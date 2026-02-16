"""Module providing freeform calibration functionality for continuous eye tracking."""

import threading
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from sklearn import linear_model as scireg


class FreeformCalibrator:
    """
    Freeform Calibration System

    Allows users to calibrate by following any movement on screen with their eyes.
    No fixed patterns required - the user starts calibration, follows the mouse cursor
    or any moving element on screen, and stops when done. Calibration can be resumed
    from where it left off if error increases.
    """

    # Calibration parameters
    PRECISION_LIMIT = 50
    PRECISION_STEP = 10
    ACCEPTANCE_RADIUS = 500
    CALIBRATION_RADIUS = 1000

    def __init__(self, calibration_radius: int = 1000) -> None:
        """
        Initialize the Freeform Calibrator.

        Args:
            calibration_radius: Radius in pixels for calibration acceptance
        """
        # Training data storage
        self.X: List[npt.NDArray[np.float64]] = []  # Eye feature vectors
        self.Y_y: List[npt.NDArray[np.float64]] = []  # Screen Y coordinates
        self.Y_x: List[npt.NDArray[np.float64]] = []  # Screen X coordinates

        # Temporary storage for accumulating points before fitting
        self.__tmp_X: List[npt.NDArray[np.float64]] = []
        self.__tmp_Y_y: List[npt.NDArray[np.float64]] = []
        self.__tmp_Y_x: List[npt.NDArray[np.float64]] = []

        # Regression models
        self.reg_x = scireg.Ridge(alpha=0.5)
        self.reg_y = scireg.Ridge(alpha=0.5)
        self.current_algorithm = "Ridge"
        self.fitted = False
        self.cv_not_set = True
        self.fixations_x = None
        self.fixations_y = None

        # Calibration state
        self.calibration_active = False
        self.calibration_paused = False
        self.points_collected = 0

        # Configuration parameters
        self.precision_limit = self.PRECISION_LIMIT
        self.precision_step = self.PRECISION_STEP
        self.acceptance_radius = int(calibration_radius / 2)
        self.calibration_radius = int(calibration_radius)

        # Threading for asynchronous model fitting
        self.lock = threading.Lock()
        self.calculation_coroutine = threading.Thread(target=self.__async_post_fit, daemon=True)
        self.fit_coroutines: List[threading.Thread] = []

    def start_calibration(self) -> None:
        """Start or resume calibration. User should follow cursor with eyes."""
        with self.lock:
            self.calibration_active = True
            self.calibration_paused = False
            print(f"[Freeform Calibration] Started. Points collected so far: {self.points_collected}")

    def pause_calibration(self) -> None:
        """Pause calibration while keeping collected data. Can be resumed later."""
        with self.lock:
            self.calibration_active = False
            self.calibration_paused = True
            print(f"[Freeform Calibration] Paused. Total points collected: {self.points_collected}")

    def stop_calibration(self) -> bool:
        """Stop calibration and finalize the model."""
        with self.lock:
            self.calibration_active = False
            self.calibration_paused = False

            if self.points_collected >= 10:  # Minimum points for meaningful calibration
                print(f"[Freeform Calibration] Stopped. Finalizing with {self.points_collected} points.")
                return True
            else:
                print(f"[Freeform Calibration] Not enough points collected ({self.points_collected}/10)")
                return False

    def add_calibration_point(
            self, eye_features: npt.NDArray[np.float64], screen_point: npt.NDArray[np.float64]
    ) -> None:
        """
        Add a calibration point during active calibration.

        Args:
            eye_features: Feature vector from eye tracking (e.g., pupil position, gaze direction)
            screen_point: Corresponding point on screen [x, y]
        """
        if not self.calibration_active:
            return

        with self.lock:
            self.__tmp_X.append(eye_features.flatten())
            self.__tmp_Y_y.append(screen_point[1])  # Y coordinate
            self.__tmp_Y_x.append(screen_point[0])  # X coordinate
            self.points_collected += 1

            # Trigger fitting when we have enough points
            if len(self.__tmp_X) >= 5:
                self.__launch_fit()

    def add_calibration_batch(
            self,
            eye_features_batch: List[npt.NDArray[np.float64]],
            screen_points_batch: List[npt.NDArray[np.float64]],
    ) -> None:
        """
        Add multiple calibration points at once.

        Args:
            eye_features_batch: List of feature vectors from eye tracking
            screen_points_batch: List of corresponding screen points
        """
        if len(eye_features_batch) != len(screen_points_batch):
            raise ValueError("Mismatched batch sizes")

        for features, point in zip(eye_features_batch, screen_points_batch):
            self.add_calibration_point(features, point)

    def predict(self, eye_features: npt.NDArray[np.float64]) -> Optional[npt.NDArray[np.float64]]:
        """
        Predict screen position based on eye features.

        Args:
            eye_features: Feature vector from eye tracking

        Returns:
            Predicted [x, y] position on screen, or None if model not fitted
        """
        if not self.fitted:
            return None

        try:
            with self.lock:
                x_pred = self.reg_x.predict([eye_features.flatten()])[0]
                y_pred = self.reg_y.predict([eye_features.flatten()])[0]
                return np.array([x_pred, y_pred])
        except Exception as e:
            print(f"[Prediction Error] {e}")
            return None

    def is_fitted(self) -> bool:
        """Check if calibration model is ready for predictions."""
        return self.fitted

    def get_calibration_quality(self) -> dict:
        """
        Get quality metrics of the current calibration.

        Returns:
            Dictionary with calibration quality metrics
        """
        return {
            "fitted": self.fitted,
            "points_collected": self.points_collected,
            "active": self.calibration_active,
            "paused": self.calibration_paused,
            "algorithm": self.current_algorithm,
        }

    def reset_calibration(self) -> None:
        """Clear all calibration data and start fresh."""
        with self.lock:
            self.X.clear()
            self.Y_x.clear()
            self.Y_y.clear()
            self.__tmp_X.clear()
            self.__tmp_Y_x.clear()
            self.__tmp_Y_y.clear()
            self.points_collected = 0
            self.fitted = False
            self.calibration_active = False
            self.calibration_paused = False
            print("[Freeform Calibration] Reset - all data cleared")

    # ==================== Private Methods ====================

    def __launch_fit(self) -> None:
        """Launch a fitting coroutine in background."""
        coroutine = threading.Thread(target=self.__async_fit, daemon=True)
        self.fit_coroutines.append(coroutine)
        coroutine.start()
        self.__join_finished()

    def __join_finished(self) -> None:
        """Wait for finished fitting coroutines."""
        for coroutine in self.fit_coroutines[:]:
            if not coroutine.is_alive():
                coroutine.join()
                self.fit_coroutines.remove(coroutine)

    def __async_fit(self) -> None:
        """Asynchronously fit the regression models."""
        try:
            with self.lock:
                if len(self.__tmp_X) == 0:
                    return

                # Combine temporary and permanent data
                fit_X = np.array(self.__tmp_X + self.X, dtype=object)
                fit_Y_y = np.array(self.__tmp_Y_y + self.Y_y)
                fit_Y_x = np.array(self.__tmp_Y_x + self.Y_x)

                # Fit regression models
                self.reg_x.fit(fit_X, fit_Y_x)
                self.reg_y.fit(fit_X, fit_Y_y)

                # Move temporary data to permanent storage
                self.X.extend(self.__tmp_X)
                self.Y_y.extend(self.__tmp_Y_y)
                self.Y_x.extend(self.__tmp_Y_x)
                self.__tmp_X.clear()
                self.__tmp_Y_y.clear()
                self.__tmp_Y_x.clear()

                self.fitted = True
                print(f"[Freeform Calibration] Model fitted with {len(self.X)} total points")

        except Exception as e:
            print(f"[Fitting Error] {e}")

    def __async_post_fit(self) -> None:
        """Post-processing after fitting (reserved for future enhancements)."""
        pass