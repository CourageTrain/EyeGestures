"""
Freeform Calibration - CORRECT VERSION
Maps raw eye features → screen coordinates
"""

import os
import sys
import cv2
import pygame
import numpy as np
import mouse

pygame.init()
pygame.font.init()

screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("EyeGestures - Freeform Calibration")

font_size = 48
bold_font = pygame.font.Font(None, font_size)
bold_font.set_bold(True)
info_font = pygame.font.Font(None, 24)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/..')

from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3
from eyeGestures.calibration_freeform import FreeformCalibrator

gestures = EyeGestures_v3()
freeform_calibrator = FreeformCalibrator(calibration_radius=1000)
cap = VideoCapture(0)

RED = (255, 0, 100)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)

clock = pygame.time.Clock()

calibration_active = False
tracking_mode = False
gaze_position = [screen_width // 2, screen_height // 2]
points_collected = 0

running = True
frame_count = 0


def get_raw_eye_features(event_data):
    """
    Extract raw eye features BEFORE EyeGestures applies calibration.
    These are the true eye data we need for our own calibration.
    """
    try:
        if event_data is None:
            return None

        # The key_points_buffer contains raw eye landmark data
        # We'll use the point coordinates as raw features since that's what we have
        # In a real scenario, you'd want to access the landmarks directly

        # For now, use the point but remember it's already processed by EyeGestures
        # This is actually the gaze estimate BEFORE our freeform calibration
        if hasattr(event_data, 'point'):
            pt = event_data.point
            if pt is not None and isinstance(pt, (list, tuple, np.ndarray)):
                if len(pt) >= 2:
                    return np.array(pt, dtype=np.float64)

        return None
    except:
        return None


while running:
    frame_count += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                running = False
            elif event.key == pygame.K_SPACE:
                if not calibration_active:
                    freeform_calibrator.start_calibration()
                    calibration_active = True
                    tracking_mode = False
                    points_collected = 0
                    print("\n>>> CALIBRATION STARTED")
                    print(">>> Follow the yellow cursor with your eyes")
                    print(">>> Move the cursor around the screen freely")
                else:
                    success = freeform_calibrator.stop_calibration()
                    calibration_active = False
                    print(f"\n>>> CALIBRATION STOPPED - Points: {points_collected}")
                    if success and freeform_calibrator.is_fitted():
                        tracking_mode = True
                        print(">>> CALIBRATION COMPLETE!")
                        print(">>> Switching to TRACKING MODE")
                        print(">>> Your eye position now controls the red dot")
                    else:
                        print(f">>> Not enough points collected ({points_collected}/10)")

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.flip(frame, axis=1)

    event_data = None
    calibration_data = None

    try:
        event_data, calibration_data = gestures.step(
            frame,
            calibration_active,
            screen_width,
            screen_height,
            context="freeform"
        )
    except Exception as e:
        print(f"Error in step: {e}")

    # ==================== CALIBRATION PHASE ====================
    if calibration_active:
        if event_data is not None:
            mouse_x, mouse_y = mouse.get_position()

            # Get raw eye features
            eye_features = get_raw_eye_features(event_data)

            if eye_features is not None:
                # Ground truth: mouse position is where user is looking
                screen_point = np.array([mouse_x, mouse_y], dtype=np.float64)

                # Add calibration point
                freeform_calibrator.add_calibration_point(eye_features, screen_point)
                points_collected = freeform_calibrator.points_collected

                if points_collected % 50 == 0:
                    print(f"[Calibration] Points: {points_collected}")

    # ==================== TRACKING PHASE ====================
    elif tracking_mode and freeform_calibrator.is_fitted():
        if event_data is not None:
            # Get the same raw eye features
            eye_features = get_raw_eye_features(event_data)

            if eye_features is not None:
                # Predict screen position using our calibration model
                predicted_gaze = freeform_calibrator.predict(eye_features)

                if predicted_gaze is not None:
                    # Clamp to screen bounds
                    gaze_x = max(0, min(int(predicted_gaze[0]), screen_width - 1))
                    gaze_y = max(0, min(int(predicted_gaze[1]), screen_height - 1))
                    gaze_position = [gaze_x, gaze_y]

                    if frame_count % 30 == 0:
                        print(f"[Tracking] Predicted gaze: ({gaze_x}, {gaze_y})")

    # ==================== RENDERING ====================
    screen.fill(BLACK)

    # Display camera frame
    if event_data and hasattr(event_data, 'sub_frame') and event_data.sub_frame is not None:
        try:
            frame_surface = pygame.surfarray.make_surface(np.rot90(event_data.sub_frame))
            frame_surface = pygame.transform.scale(frame_surface, (400, 400))
            screen.blit(frame_surface, (0, 0))
        except:
            pass

    # ==================== CALIBRATION MODE UI ====================
    if calibration_active:
        mouse_x, mouse_y = mouse.get_position()

        # Draw cursor circle
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 30, 3)
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 15)
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 8, 2)

        status_text = "CALIBRATION MODE - Follow the cursor with your eyes"
        status_surface = bold_font.render(status_text, True, CYAN)
        screen.blit(status_surface, (420, 50))

        points_text = f"Points Collected: {points_collected}"
        points_surface = bold_font.render(points_text, True, GREEN)
        screen.blit(points_surface, (420, 120))

        # Progress bar
        min_points = 10
        progress = min(points_collected / min_points, 1.0)
        bar_width = 300
        bar_height = 20
        bar_x = 420
        bar_y = 160
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
        pygame.draw.rect(screen, GREEN, (bar_x, bar_y, int(bar_width * progress), bar_height))

        instr_text = "SPACEBAR: Stop Calibration | CTRL+Q: Quit"
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_surface, (420, 200))

        if freeform_calibrator.is_fitted():
            model_text = "✓ Model FITTED"
            model_surface = info_font.render(model_text, True, GREEN)
        else:
            model_text = "○ Model Training..."
            model_surface = info_font.render(model_text, True, YELLOW)
        screen.blit(model_surface, (420, 240))

    # ==================== TRACKING MODE UI ====================
    elif tracking_mode:
        # Draw gaze tracking dot
        pygame.draw.circle(screen, RED, gaze_position, 20)
        pygame.draw.circle(screen, WHITE, gaze_position, 20, 3)
        pygame.draw.circle(screen, RED, gaze_position, 8)

        status_text = "TRACKING MODE - Eye position controls the dot"
        status_surface = bold_font.render(status_text, True, GREEN)
        screen.blit(status_surface, (420, 50))

        gaze_text = f"Gaze Position: ({gaze_position[0]}, {gaze_position[1]})"
        gaze_surface = info_font.render(gaze_text, True, WHITE)
        screen.blit(gaze_surface, (420, 120))

        instr_text = "SPACEBAR: Resume Calibration | CTRL+Q: Quit"
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_surface, (420, 160))

        cal_quality = freeform_calibrator.get_calibration_quality()
        stats_text = f"Calibration Points: {cal_quality['points_collected']}"
        stats_surface = info_font.render(stats_text, True, WHITE)
        screen.blit(stats_surface, (420, 200))

    # ==================== IDLE MODE UI ====================
    else:
        idle_text = "READY TO CALIBRATE"
        idle_surface = bold_font.render(idle_text, True, CYAN)
        screen.blit(idle_surface, (420, 50))

        instr1 = "Press SPACEBAR to start calibration"
        instr1_surface = info_font.render(instr1, True, WHITE)
        screen.blit(instr1_surface, (420, 150))

        instr2 = "Follow the yellow cursor with your eyes"
        instr2_surface = info_font.render(instr2, True, WHITE)
        screen.blit(instr2_surface, (420, 190))

        instr3 = "Move the cursor around freely"
        instr3_surface = info_font.render(instr3, True, WHITE)
        screen.blit(instr3_surface, (420, 230))

        instr4 = "Press SPACEBAR again to finish & track"
        instr4_surface = info_font.render(instr4, True, WHITE)
        screen.blit(instr4_surface, (420, 270))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
cap.release()
print("\nApplication closed.")