import numpy as np


class KalmanFilter:
    """
    2D constant-velocity Kalman filter.

    State vector:
        x = [pos_x, pos_y, vel_x, vel_y]^T

    Measurement vector:
        z = [measured_x, measured_y]^T
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        initial_covariance: float = 1000.0,
    ):
        self.dt = float(dt)

        # State vector [x, y, vx, vy]^T
        self.x = np.zeros((4, 1), dtype=np.float64)

        # State transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float64) * float(process_noise)

        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float64) * float(measurement_noise)

        # Estimate covariance
        self.P = np.eye(4, dtype=np.float64) * float(initial_covariance)

        # Identity matrix
        self.I = np.eye(4, dtype=np.float64)

        self.initialized = False

    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        """Initialize the filter state."""
        self.x = np.array([[x], [y], [vx], [vy]], dtype=np.float64)
        self.initialized = True

    def predict(self) -> tuple[float, float]:
        """
        Predict the next state using the motion model.
        Returns predicted position (x, y).
        """
        if not self.initialized:
            raise RuntimeError("Kalman filter is not initialized.")

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, measured_x: float, measured_y: float) -> tuple[float, float]:
        """
        Correct the prediction using a measured position.
        Returns corrected position (x, y).
        """
        if not self.initialized:
            self.initialize(measured_x, measured_y)
            return float(self.x[0, 0]), float(self.x[1, 0])

        z = np.array([[measured_x], [measured_y]], dtype=np.float64)

        # Innovation / residual
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state estimate
        self.x = self.x + (K @ y)

        # Updated covariance
        self.P = (self.I - K @ self.H) @ self.P

        return float(self.x[0, 0]), float(self.x[1, 0])

    def get_state(self) -> dict:
        """Return the current estimated state."""
        return {
            "x": float(self.x[0, 0]),
            "y": float(self.x[1, 0]),
            "vx": float(self.x[2, 0]),
            "vy": float(self.x[3, 0]),
        }

    def set_process_noise(self, value: float) -> None:
        self.Q = np.eye(4, dtype=np.float64) * float(value)

    def set_measurement_noise(self, value: float) -> None:
        self.R = np.eye(2, dtype=np.float64) * float(value)

    def reset(self) -> None:
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1000.0
        self.initialized = False