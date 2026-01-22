import os
import sys
import glob
import pygame


# ----------------------------
# Config
# ----------------------------
SCREEN_W, SCREEN_H = 960, 540
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

CHAR_NAME = "dude"
BG_NAME = "street_day"

MOVE_SPEED_PX_PER_SEC = 260.0

# Pygame init
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Video-Frames -> Pygame Sprites Prototype")
clock = pygame.time.Clock()


# ----------------------------
# Helpers
# ----------------------------
def load_frame_folder(folder: str):
    """
    Loads all frame_*.png/jpg in folder (sorted) as convert_alpha() surfaces.
    """
    if not os.path.isdir(folder):
        return []

    files = sorted(
        glob.glob(os.path.join(folder, "*.png")) +
        glob.glob(os.path.join(folder, "*.jpg")) +
        glob.glob(os.path.join(folder, "*.jpeg"))
    )
    frames = []
    for fp in files:
        try:
            img = pygame.image.load(fp).convert_alpha()
            frames.append(img)
        except Exception as e:
            print(f"[warn] failed loading {fp}: {e}")
    return frames


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Animation:
    def __init__(self, frames, fps=12.0):
        self.frames = frames or []
        self.fps = float(fps)
        self.t = 0.0
        self.idx = 0

    def reset(self):
        self.t = 0.0
        self.idx = 0

    def update(self, dt):
        if len(self.frames) <= 1 or self.fps <= 0:
            return
        self.t += dt
        frame_advance = int(self.t * self.fps)
        if frame_advance > 0:
            self.t -= frame_advance / self.fps
            self.idx = (self.idx + frame_advance) % len(self.frames)

    def get(self):
        if not self.frames:
            return None
        return self.frames[self.idx]


class Character:
    def __init__(self, name: str, base_dir: str, fps=12.0):
        self.name = name
        self.base_dir = base_dir
        self.fps = fps

        # runtime state
        self.x = SCREEN_W * 0.5
        self.y = SCREEN_H * 0.7

        self.vx = 0.0
        self.vy = 0.0

        self.angle = "front"   # front/back/left/right
        self.anim = "idle"     # idle/walk
        self.last_angle = self.angle

        # cache animations: anims[angle][anim] = Animation(...)
        self.anims = {}
        self._load_all()

    def _load_all(self):
        """
        Loads whatever exists on disk. Missing combos fallback gracefully.
        """
        angles_dir = os.path.join(self.base_dir, "angles")
        if not os.path.isdir(angles_dir):
            raise RuntimeError(f"Missing angles dir: {angles_dir}")

        for angle in sorted(os.listdir(angles_dir)):
            angle_path = os.path.join(angles_dir, angle)
            if not os.path.isdir(angle_path):
                continue
            self.anims.setdefault(angle, {})
            for anim in sorted(os.listdir(angle_path)):
                anim_path = os.path.join(angle_path, anim)
                if not os.path.isdir(anim_path):
                    continue
                frames = load_frame_folder(anim_path)
                if frames:
                    self.anims[angle][anim] = Animation(frames, fps=self.fps)

        if not self.anims:
            raise RuntimeError(f"No animations found under: {angles_dir}")

    def _pick_angle_from_velocity(self):
        eps = 1e-3
        # Prefer the dominant axis for "angle"
        if abs(self.vx) > abs(self.vy):
            if self.vx > eps:
                return "right"
            if self.vx < -eps:
                return "left"
        else:
            if self.vy > eps:
                return "front"   # down
            if self.vy < -eps:
                return "back"    # up
        return self.last_angle

    def _has(self, angle, anim):
        return angle in self.anims and anim in self.anims[angle]

    def _fallback_angle(self, desired):
        # If desired missing, try last_angle, then any available
        if desired in self.anims:
            return desired
        if self.last_angle in self.anims:
            return self.last_angle
        return next(iter(self.anims.keys()))

    def _fallback_anim(self, angle, desired):
        # If desired missing, try idle, then any available
        if self._has(angle, desired):
            return desired
        if self._has(angle, "idle"):
            return "idle"
        return next(iter(self.anims[angle].keys()))

    def update(self, dt):
        # movement
        self.x += self.vx * dt
        self.y += self.vy * dt

        self.x = clamp(self.x, 0, SCREEN_W)
        self.y = clamp(self.y, 0, SCREEN_H)

        moving = (abs(self.vx) + abs(self.vy)) > 1e-3
        desired_anim = "walk" if moving else "idle"
        desired_angle = self._pick_angle_from_velocity()

        # resolve missing folders gracefully
        angle = self._fallback_angle(desired_angle)
        anim = self._fallback_anim(angle, desired_anim)

        # change anim? reset animation timer
        if angle != self.angle or anim != self.anim:
            # reset the new one
            if angle in self.anims and anim in self.anims[angle]:
                self.anims[angle][anim].reset()

        self.angle = angle
        self.anim = anim
        self.last_angle = angle

        # update animation
        self.anims[self.angle][self.anim].update(dt)

    def draw(self, surf):
        frame = self.anims[self.angle][self.anim].get()
        if not frame:
            return

        # draw centered on feet-ish (bottom-center)
        rect = frame.get_rect()
        rect.midbottom = (int(self.x), int(self.y))
        surf.blit(frame, rect)


class Background:
    def __init__(self, folder: str, fps=12.0):
        self.frames = load_frame_folder(folder)
        self.anim = Animation(self.frames, fps=fps) if self.frames else None

    def update(self, dt):
        if self.anim:
            self.anim.update(dt)

    def draw(self, surf):
        if not self.anim:
            surf.fill((20, 20, 20))
            return

        frame = self.anim.get()
        if not frame:
            surf.fill((20, 20, 20))
            return

        # "cover" scale to fill screen without distortion (simple)
        fw, fh = frame.get_width(), frame.get_height()
        sw, sh = surf.get_width(), surf.get_height()

        scale = max(sw / max(1, fw), sh / max(1, fh))
        nw, nh = int(fw * scale), int(fh * scale)
        scaled = pygame.transform.smoothscale(frame, (nw, nh))

        # center crop
        x = (nw - sw) // 2
        y = (nh - sh) // 2
        surf.blit(scaled, (-x, -y))


# ----------------------------
# Asset paths
# ----------------------------
char_dir = os.path.join(ASSETS_DIR, "characters", CHAR_NAME)
bg_dir = os.path.join(ASSETS_DIR, "backgrounds", BG_NAME)

if not os.path.isdir(char_dir):
    print(f"[error] Missing character dir: {char_dir}")
    sys.exit(1)

player = Character(CHAR_NAME, char_dir, fps=12.0)
bg = Background(bg_dir, fps=12.0)


# ----------------------------
# Main loop
# ----------------------------
running = True
while running:
    dt = clock.tick(60) / 1000.0  # seconds

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # WASD movement
    vx = 0.0
    vy = 0.0
    if keys[pygame.K_a]:
        vx -= MOVE_SPEED_PX_PER_SEC
    if keys[pygame.K_d]:
        vx += MOVE_SPEED_PX_PER_SEC
    if keys[pygame.K_w]:
        vy -= MOVE_SPEED_PX_PER_SEC
    if keys[pygame.K_s]:
        vy += MOVE_SPEED_PX_PER_SEC

    # normalize diagonal
    if vx != 0.0 and vy != 0.0:
        inv = 0.70710678
        vx *= inv
        vy *= inv

    player.vx = vx
    player.vy = vy

    bg.update(dt)
    player.update(dt)

    bg.draw(screen)
    player.draw(screen)

    # debug overlay
    font = pygame.font.SysFont(None, 22)
    txt = font.render(f"angle={player.angle} anim={player.anim}", True, (255, 255, 255))
    screen.blit(txt, (12, 12))

    pygame.display.flip()

pygame.quit()
