"""RebotArmEndPose — 末端位姿控制器（IK + 轨迹规划）。

同时支持两种运动模式：

  - ``move_to_ik(...)``   即时 IK 求解，关节角度一步到位。
  - ``move_to_traj(...)`` SE(3) 测地线规划 + CLIK 跟踪，平滑轨迹运动。

arm 部分支持两种控制模式（由 ``arm_control_mode`` 选择）：

  - ``"posvel"``（默认）：位置+速度模式，电机内部 PID 闭环。
  - ``"mit"``           ：MIT 阻抗控制模式，主机下发 pos/vel/kp/kd/tau 五元组。

控制循环中按组发送：rebotarm.arm.send_pos_vel() → rebotarm.gripper.send_mit()
（posvel 模式），或 rebotarm.arm.send_mit() → rebotarm.gripper.send_mit()（mit 模式）。

使用示例::
----
    from reBotArm_control_py.controllers import RebotArmEndPose

    rebotarm = RebotArm()

    # POS_VEL 模式（默认）
    ctrl = RebotArmEndPose(rebotarm, arm_control_mode="posvel")
    ctrl.start()
    ctrl.move_to_ik(x=0.3, y=0.0, z=0.3)
    ctrl.move_to_traj(x=0.3, y=0.0, z=0.3, duration=2.0)
    ctrl.end()
----
    from reBotArm_control_py.controllers import RebotArmEndPose

    rebotarm = RebotArm()

    # MIT 模式
    ctrl_mit = RebotArmEndPose(rebotarm, arm_control_mode="mit")
    ctrl_mit.start()
    ctrl_mit.move_to_ik(x=0.3, y=0.0, z=0.3)
    ctrl_mit.move_to_traj(x=0.3, y=0.0, z=0.3, duration=2.0)
    ctrl_mit.end()

上下文管理器::

    with RebotArmEndPose(rebotarm, arm_control_mode="mit") as ctrl:
        ctrl.move_to_ik(x=0.3, y=0.0, z=0.3)
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from ..dynamics import compute_generalized_gravity
from ..kinematics import (
    compute_fk,
    pos_rot_to_se3,
    get_end_effector_frame_id,
    load_robot_model,
    pad_q_for_model,
)
from ..kinematics.inverse_kinematics import (
    solve_ik,
    solve_ik_with_retry,
    IKParams as TrajIKParams,
)
from ..trajectory import (
    TrajProfile,
    TrajPlanParams,
    IKParams as ClikIKParams,
    plan_cartesian_geodesic_trajectory,
    track_trajectory,
)
from ..actuator import RebotArm


class RebotArmEndPose:

    def __init__(
        self,
        rebotarm: RebotArm,
        dt: float = 0.01,
        profile: TrajProfile = TrajProfile.MIN_JERK,
        arm_control_mode: str = "posvel",
        use_gravity_ff: bool = True,
    ) -> None:
        if arm_control_mode not in ("mit", "posvel"):
            raise ValueError("arm_control_mode must be 'mit' or 'posvel'")
        self._arm_control_mode = arm_control_mode
        self._use_gravity_ff = use_gravity_ff
        self.rebotarm = rebotarm
        self._arm_group = rebotarm.groups.get("arm", None)
        self._gripper_group = rebotarm.groups.get("gripper", None)
        self._has_gripper = rebotarm.has_gripper

        if self._arm_group is None:
            raise ValueError("配置中缺少 arm 组，请检查 groups 配置")

        self._n = self._arm_group.num_joints
        self._dt = dt
        self._model = load_robot_model()
        self._end_frame_id = get_end_effector_frame_id(self._model)
        self._data = self._model.createData()

        self._traj_params = TrajPlanParams(dt=dt, profile=profile)
        self._ik_solver_params = TrajIKParams(
            max_iter=200, tolerance=2e-3, step_size=0.5, damping=1e-6,
        )
        self._ik_max_retries = 150
        self._clik_params = ClikIKParams(
            max_iter=50, tolerance=1e-3, damping=1e-6, step_size=0.8,
        )
        # Sicherheitsgrenze: CLIK kann an Singularitaeten/Konfigurations-
        # mehrdeutigkeiten stellenweise nicht konvergieren -- ohne Pruefung
        # wuerde der Arm dann zwischen zwei Punkten "schnappen" (siehe
        # move_to_traj()). 5 rad/s liegt weit ueber normalen Bewegungen
        # (~1 rad/s beobachtet), faengt aber echte Spruenge (>50 rad/s) ab.
        self._max_joint_speed = 5.0

        self._q_target: np.ndarray = np.zeros(self._n)
        self._qd_target: np.ndarray = np.zeros(self._n)
        self._gripper_target: float = 0.0
        self._gripper_force_mode: bool = False
        self._gripper_tau: float = 0.0
        self._gripper_kd_force: float = 0.3
        self._running = False

        self._traj: list[np.ndarray] = []
        self._moving = False
        self._send_thread: Optional[threading.Thread] = None
        self._stop_send = threading.Event()

        self._home_vel: float = 0.5
        self._vlim_override: Optional[np.ndarray] = None

    # ── 生命周期 ───────────────────────────────────────────────────────────

    def start(self) -> None:
        self.rebotarm.connect()
        if self._arm_group:
            if self._arm_control_mode == "mit":
                self._arm_group.mode_mit(
                    kp=self._arm_group._mit_kp,
                    kd=self._arm_group._mit_kd,
                )
            else:
                self._arm_group.mode_pos_vel()
            self._arm_group.enable()
        if self._has_gripper:
            self._gripper_group.mode_mit()
            self._gripper_group.enable()

        # Ist-Position uebernehmen, bevor der Regelkreis startet -- sonst wuerde er
        # sofort mit vollem kp/kd Richtung der bei __init__ initialisierten
        # Nullstellung ziehen (gefaehrlicher Sprung, unabhaengig vom ersten move_to_*).
        # Direkt nach connect()/enable() ist noch kein echtes Feedback da (get_state()
        # liefert dann Nullen als Platzhalter) -- daher mehrfach pollen, bis reale
        # Werte ankommen.
        pos = np.zeros(self._n + (1 if self._has_gripper else 0))
        for _ in range(25):
            pos, _, _ = self.rebotarm.get_state()
            if np.any(pos != 0.0):
                break
            time.sleep(0.02)
        self._q_target = pos[: self._n].copy()
        if self._has_gripper:
            self._gripper_target = float(pos[-1])

        self.rebotarm.start_control_loop(self._loop_cb)
        self._running = True

    def end(self) -> None:
        if not self._running:
            return
        self.safe_home()
        self.rebotarm.disconnect()
        self._running = False

    def __enter__(self) -> "RebotArmEndPose":
        return self

    def __exit__(self, *args) -> None:
        self.end()

    # ── 公共 API ───────────────────────────────────────────────────────────

    def set_gripper_target(self, pos: float) -> None:
        self._gripper_target = float(pos)

    def open_gripper(self, tau: float = -1.0) -> None:
        """Kraftregelung (kp=0): konstantes Oeffnungs-Drehmoment tau [Nm] (kalibriert: negativ=oeffnen)."""
        if self._has_gripper:
            self._gripper_force_mode = True
            self._gripper_tau = float(tau)

    def close_gripper(self, tau: float = 1.0) -> None:
        """Kraftregelung (kp=0): konstantes Greif-Drehmoment tau [Nm] (kalibriert: positiv=schliessen)."""
        if self._has_gripper:
            self._gripper_force_mode = True
            self._gripper_tau = float(tau)

    def safe_home(
        self,
        max_vel: float = 0.5,
        send_freq: float = 50.0,
        settle_thresh: float = 0.01,
        timeout: float = 15.0,
    ) -> None:
        if not self._running:
            return

        q_curr, _, _ = self.rebotarm.get_state()
        q_curr = q_curr[: self._n]
        q_start = q_curr.copy()

        home_pos = np.zeros(self._n)
        q_err = np.abs(home_pos - q_start)
        max_err = float(np.max(q_err))
        if max_err < 0.01:
            return

        t_ramp = max_err / max_vel
        t_total = t_ramp * 2.0
        dt_send = 1.0 / send_freq
        num_steps = max(2, int(t_total / dt_send))

        t = np.linspace(0, t_total, num_steps)
        traj = np.zeros((num_steps, self._n))
        for i in range(self._n):
            err_i = home_pos[i] - q_start[i]
            s = t / t_total
            # 最小jerk (minimum jerk) 轨迹:
            #   q(s) = q0 + Δq * (10s³ - 15s⁴ + 6s⁵)
            # 速度: v(s) = Δq/t_total * (30s² - 60s³ + 30s⁴) → 在 s=0 和 s=1 处均为零
            traj[:, i] = q_start[i] + err_i * (10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5)

        interval = t_total / num_steps if num_steps > 0 else dt_send
        deadline = time.monotonic() + timeout
        self._vlim_override = np.full(self._n, max_vel, dtype=np.float64)
        for i in range(num_steps):
            if time.monotonic() > deadline:
                print("[safe_home] 轨迹发送超时")
                break
            self._q_target[:] = traj[i]
            time.sleep(interval)

        self._q_target[:] = 0.0
        settle_deadline = time.monotonic() + 3.0
        while time.monotonic() < settle_deadline:
            q_now, _, _ = self.rebotarm.get_state()
            if np.max(np.abs(q_now[: self._n])) < settle_thresh:
                break
            time.sleep(self._dt)
        self._vlim_override = None

    def move_to_ik(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> bool:
        if not self._running:
            return False

        q_curr, _, _ = self.rebotarm.get_state()
        q_curr = pad_q_for_model(self._model, q_curr, self._n)
        T_target = pos_rot_to_se3(
            np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw,
        )

        result = solve_ik_with_retry(
            self._model, self._data, self._end_frame_id,
            T_target, q_curr, self._ik_solver_params,
            max_retries=self._ik_max_retries,
        )
        if not result.success:
            print(f"[RebotArmEndPose/IK] IK 未收敛  err={result.error:.3e}")
            return False

        self._q_target = result.q[:self._n].copy()
        return True

    def move_to_traj(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        if not self._running:
            return False

        q_start, _, _ = self.rebotarm.get_state()
        q_start = pad_q_for_model(self._model, q_start, self._n)

        T_target = pos_rot_to_se3(
            np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw,
        )

        ik_result = solve_ik_with_retry(
            self._model, self._data, self._end_frame_id,
            # q_start.copy(): solve_ik_with_retry ueberschreibt q_seed in-place
            # mit der Loesung -- ohne copy() wuerde q_start selbst (unten fuer
            # T_start gebraucht) auf den Zielwert ueberschrieben und die
            # gesamte Geodaete auf einen Punkt kollabieren.
            T_target, q_start.copy(), self._ik_solver_params,
            max_retries=self._ik_max_retries,
        )
        if not ik_result.success:
            print(f"[RebotArmEndPose/Traj] IK 失败  err={ik_result.error:.4f}")
            return False

        q_end = ik_result.q
        q_end_padded = pad_q_for_model(self._model, q_end, self._n)

        T_start = compute_fk(self._model, q_start)[2]
        T_end = compute_fk(self._model, q_end_padded)[2]

        if duration <= 0:
            dist = float(np.linalg.norm(T_target.translation() - T_start.translation()))
            duration = max(1.0, dist / 0.1)

        cart_traj = plan_cartesian_geodesic_trajectory(
            T_start, T_end, duration, self._traj_params,
        )

        joint_traj = track_trajectory(
            self._model, self._end_frame_id,
            cart_traj.trajectory, q_start, self._clik_params,
            null_gain=0.1,
        )
        if not joint_traj:
            print("[RebotArmEndPose/Traj] 轨迹为空")
            return False

        pts = [pt.q[: self._n].copy() for pt in joint_traj]

        n_fail = sum(1 for pt in joint_traj if not pt.ik_success)
        max_step = max(
            (float(np.abs(pts[i + 1] - pts[i]).max()) for i in range(len(pts) - 1)),
            default=0.0,
        )
        max_speed = max_step / self._traj_params.dt
        if max_speed > self._max_joint_speed:
            print(f"[RebotArmEndPose/Traj] Sicherheitsabbruch: Gelenksprung "
                  f"{max_speed:.2f} rad/s (> {self._max_joint_speed} rad/s, "
                  f"nicht_konvergiert={n_fail}/{len(pts)}) -- CLIK vermutlich an "
                  f"Singularitaet/Mehrdeutigkeit gescheitert. Bewegung wird NICHT gesendet.")
            return False

        self._stop_send.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=5.0)

        self._traj = pts
        self._moving = True
        self._stop_send.clear()
        self._send_thread = threading.Thread(
            target=self._send_loop, args=(duration,), daemon=True,
        )
        self._send_thread.start()
        return True

    def move_to_q_traj(self, q_target: np.ndarray, duration: float = 2.0) -> bool:
        """Faehrt direkt zu einer geteachten Gelenkkonfiguration (Min-Jerk, Gelenkraum).

        Im Gegensatz zu move_to_traj() wird KEINE IK geloest -- q_target ist die beim
        Teachen real gefahrene, damit garantiert erreichbare Konfiguration. Dadurch
        entfallen CLIK-Mehrdeutigkeiten/Singularitaeten zwischen weit auseinander
        liegenden Posen komplett (siehe move_to_traj()-Sicherheitsabbruch).
        """
        if not self._running:
            return False

        q_start, _, _ = self.rebotarm.get_state()
        q_start = pad_q_for_model(self._model, q_start, self._n)
        q_target = pad_q_for_model(self._model, np.asarray(q_target, dtype=float), self._n)

        dt = self._traj_params.dt
        n_pts = max(2, int(duration / dt))
        s = np.linspace(0.0, 1.0, n_pts)
        blend = 10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5  # Min-Jerk (wie safe_home())
        pts = [q_start + (q_target - q_start) * b for b in blend]

        self._stop_send.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=5.0)

        self._traj = pts
        self._moving = True
        self._stop_send.clear()
        self._send_thread = threading.Thread(
            target=self._send_loop, args=(duration,), daemon=True,
        )
        self._send_thread.start()
        return True

    # ── 控制循环 ───────────────────────────────────────────────────────────

    def _loop_cb(self, _: RebotArm, dt: float) -> None:
        if self._arm_group:
            if self._arm_control_mode == "mit":
                tau_ff = np.zeros(self._n)
                if self._use_gravity_ff:
                    q_now = self._arm_group.get_positions(request_feedback=False)
                    q_now = pad_q_for_model(self._model, q_now, self._n)
                    tau_ff = compute_generalized_gravity(self._model, q_now, self._data)[: self._n]
                    tau_ff[1] *= 1.55  # joint2 额外补偿
                    tau_ff[2] *= 1.55  # joint3 额外补偿
                    
                self._arm_group.send_mit(
                    self._q_target,
                    vel=self._qd_target,
                    kp=self._arm_group._mit_kp,
                    kd=self._arm_group._mit_kd,
                    tau=tau_ff,
                )
            else:
                vlim = (
                    self._vlim_override
                    if self._vlim_override is not None
                    else self._arm_group._pv_vlim
                )
                self._arm_group.send_pos_vel(self._q_target, vlim=vlim)
        if self._has_gripper:
            if self._gripper_force_mode:
                # Kraftregelung: kp=0 -> Zielposition irrelevant, konstantes Drehmoment.
                self._gripper_group.send_mit(
                    np.array([self._gripper_target]),
                    kp=np.zeros(1),
                    kd=np.full(1, self._gripper_kd_force),
                    tau=np.array([self._gripper_tau]),
                )
            else:
                self._gripper_group.send_mit(
                    np.array([self._gripper_target]),
                    kp=self._gripper_group._mit_kp,
                    kd=self._gripper_group._mit_kd,
                )

    # ── 轨迹发送线程 ──────────────────────────────────────────────────────

    def _send_loop(self, duration: float) -> None:
        n = len(self._traj)
        interval = duration / n if n > 0 else self._dt
        for i in range(n):
            if self._stop_send.is_set():
                return
            self._q_target[:] = self._traj[i]
            time.sleep(interval)
        self._moving = False
