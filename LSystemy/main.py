import matplotlib.pyplot as plt
import math
from PIL import Image
import io
import os
from tqdm import tqdm
import time

class LSystem2D:
    def __init__(self, axiom, rules, angle=25, start_pos=(0, 0), start_angle=0, 
                 step_length=1, iterations=4, filename="lsystem_2d.gif"):
        self.axiom, self.rules, self.angle = axiom, rules, angle
        self.start_pos, self.start_angle = start_pos, start_angle
        self.step_length, self.max_iterations = step_length, iterations
        self.filename = filename
        self.drawing_chars = set("ABCDEFGHIJKLMNOPQRSTU0123456789")
        self.moving_chars = set("abcdefghijklmnopqrstu")
        self.ignored_chars = set("VWXYZvwxyz")
        self.words = self._generate_all_words()
        self.frames = []
        self.max_dimensions = self.calculate_max_dimensions()
    
    def _generate_all_words(self):
        words = [self.axiom]
        for _ in range(self.max_iterations): words.append(self.generate_next_word(words[-1]))
        return words
    
    def generate_next_word(self, word): return ''.join(self.rules.get(char, char) for char in word)
    
    def get_word_at_iteration(self, iteration):
        return self.words[iteration] if 0 <= iteration < len(self.words) else None
    
    def calculate_max_dimensions(self):
        all_min_x, all_max_x = float('inf'), float('-inf')
        all_min_y, all_max_y = float('inf'), float('-inf')
        
        for i in range(len(self.words)):
            bounds = self._get_drawing_bounds(i)
            if bounds:
                min_x, max_x, min_y, max_y = bounds
                all_min_x, all_max_x = min(all_min_x, min_x), max(all_max_x, max_x)
                all_min_y, all_max_y = min(all_min_y, min_y), max(all_max_y, max_y)
        
        if all_min_x == float('inf'): return (-10, 10, -10, 10)
        
        width, height = all_max_x - all_min_x, all_max_y - all_min_y
        max_size = max(width, height)
        center_x, center_y = (all_min_x + all_max_x) / 2, (all_min_y + all_max_y) / 2
        padding = max_size * 0.1
        max_size_with_padding = max_size + 2 * padding
        
        return (center_x - max_size_with_padding / 2, center_x + max_size_with_padding / 2,
                center_y - max_size_with_padding / 2, center_y + max_size_with_padding / 2)
    
    def _get_drawing_bounds(self, iteration):
        lines_x, lines_y, colors = self.simulate_drawing(iteration)
        if not lines_x or not lines_y: return None
        valid_x = [x for x in lines_x if x is not None]
        valid_y = [y for y in lines_y if y is not None]
        if not valid_x or not valid_y: return None
        return min(valid_x), max(valid_x), min(valid_y), max(valid_y)
    
    def _process_turtle_commands(self, word):
        stack, angle, pos = [], self.start_angle, self.start_pos
        lines_x, lines_y, colors = [], [], []
        branch_level, max_branch_level = 0, 0
        
        for char in word:
            if char in self.drawing_chars:
                new_x = pos[0] + self.step_length * math.cos(math.radians(angle))
                new_y = pos[1] + self.step_length * math.sin(math.radians(angle))
                lines_x.extend([pos[0], new_x, None])
                lines_y.extend([pos[1], new_y, None])
                green_shade = 0.3 + 0.3 * (branch_level / (max_branch_level + 1)) if max_branch_level > 0 else 0.4
                colors.append((0.1, green_shade, 0.1))
                pos = (new_x, new_y)
            elif char in self.moving_chars:
                pos = (pos[0] + self.step_length * math.cos(math.radians(angle)),
                       pos[1] + self.step_length * math.sin(math.radians(angle)))
            elif char == "+": angle = (angle + self.angle) % 360
            elif char == "-": angle = (angle - self.angle) % 360
            elif char == "|": angle = (angle + 180) % 360
            elif char == "[":
                stack.append((pos, angle, branch_level))
                branch_level += 1
                max_branch_level = max(max_branch_level, branch_level)
            elif char == "]" and stack: pos, angle, branch_level = stack.pop()
        return lines_x, lines_y, colors
    
    def simulate_drawing(self, iteration):
        word = self.get_word_at_iteration(iteration)
        return self._process_turtle_commands(word) if word is not None else ([], [], [])
    
    def draw_iteration(self, iteration, ax=None, color='g'):
        if ax is None: fig, ax = plt.subplots(figsize=(10, 10))
        word = self.get_word_at_iteration(iteration)
        if word is None:
            print(f"Iterace {iteration} není k dispozici.")
            return None, None
            
        lines_x, lines_y, colors = self._process_turtle_commands(word)
        
        if lines_x and lines_y:
            segments_x, segments_y = [], []
            current_segment_x, current_segment_y = [], []
            current_color_idx = 0
            
            for i in range(len(lines_x)):
                if lines_x[i] is None or lines_y[i] is None:
                    if current_segment_x:
                        segments_x.append(current_segment_x)
                        segments_y.append(current_segment_y)
                        current_segment_x, current_segment_y = [], []
                        current_color_idx += 1
                else:
                    current_segment_x.append(lines_x[i])
                    current_segment_y.append(lines_y[i])
            
            if current_segment_x:
                segments_x.append(current_segment_x)
                segments_y.append(current_segment_y)
            
            for i in range(len(segments_x)):
                color_idx = min(i, len(colors) - 1) if colors else 0
                if colors and color_idx < len(colors):
                    ax.plot(segments_x[i], segments_y[i], color=colors[color_idx], linewidth=1)
                else: ax.plot(segments_x[i], segments_y[i], f'{color}-', linewidth=1)
                    
        ax.set_title(f"Iterace {iteration}")
        ax.set_xlim(self.max_dimensions[0], self.max_dimensions[1])
        ax.set_ylim(self.max_dimensions[2], self.max_dimensions[3])
        ax.set_aspect('equal')
        return lines_x, lines_y
    
    def save_last_iteration_png(self):
        last_iteration = len(self.words) - 1
        fig, ax = plt.subplots(figsize=(10, 10))
        self.draw_iteration(last_iteration, ax)
        png_filename = os.path.splitext(self.filename)[0] + ".png"
        plt.savefig(png_filename, format='png', bbox_inches='tight', dpi=800)
        plt.close(fig)
        print(f"Poslední iterace uložena jako: {png_filename}")
        return png_filename
    
    def create_gif(self):
        start_time = time.time()
        print(f"Generování GIF pro {self.filename}...")
        
        for i in tqdm(range(len(self.words))):
            fig, ax = plt.subplots(figsize=(10, 10))
            self.draw_iteration(i, ax)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            self.frames.append(Image.open(buf).copy())
            plt.close(fig)
        
        if self.frames:
            self.frames[0].save(self.filename, format='GIF', append_images=self.frames[1:],
                save_all=True, duration=1000, loop=0)
            self.save_last_iteration_png()
            duration = time.time() - start_time
            print(f"GIF uložen jako: {self.filename} (doba generování: {duration:.2f}s)")
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(self.frames[0])
            plt.title("První iterace")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(self.frames[-1])
            plt.title(f"Poslední iterace ({len(self.words)-1})")
            plt.axis('off')
            plt.suptitle(f"L-systém: {os.path.basename(self.filename)}")
            plt.close()
        else: print(f"Nepodařilo se vytvořit žádné snímky pro GIF.")
        return self.filename
    
class LSystem3D:
    def __init__(self, axiom, rules, angle=25, start_pos=(0, 0, 0), start_heading=(0, 0, 1),
                 start_left=(1, 0, 0), step_length=10, iterations=5, filename="lsystem_3d.gif"):
        self.axiom, self.rules, self.angle = axiom, rules, angle
        self.start_pos, self.step_length = start_pos, step_length
        self.start_heading = self._normalize(start_heading)
        self.start_left = self._normalize(start_left)
        self.max_iterations, self.filename = iterations, filename
        self.drawing_chars = set("ABCDEFGHIJKLMNOPQRSTU0123456789")
        self.moving_chars = set("abcdefghijklmnopqrstu")
        self.ignored_chars = set("VWXYZvwxyz")
        self.words = self._generate_all_words()
        self.frames = []
        
    def _normalize(self, vector):
        length = math.sqrt(sum(x*x for x in vector))
        return tuple(x/length for x in vector) if length > 0 else vector
    
    def _cross_product(self, a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    
    def _generate_all_words(self):
        words = [self.axiom]
        for _ in range(self.max_iterations): words.append(self.generate_next_word(words[-1]))
        return words
    
    def generate_next_word(self, word): return ''.join(self.rules.get(char, char) for char in word)
    
    def get_word_at_iteration(self, iteration):
        return self.words[iteration] if 0 <= iteration < len(self.words) else None
    
    def _process_turtle_commands(self, word):
        stack, pos, heading, left = [], self.start_pos, self.start_heading, self.start_left
        up = self._cross_product(heading, left)
        points, colors, segments = [], [], []
        branch_level, max_branch_level = 0, 0
        
        for char in word:
            if char in self.drawing_chars:
                old_pos = pos
                new_pos = (pos[0]+self.step_length*heading[0], pos[1]+self.step_length*heading[1], 
                           pos[2]+self.step_length*heading[2])
                segments.append((old_pos, new_pos))
                green_shade = 0.3 + 0.3 * (branch_level / (max_branch_level + 1)) if max_branch_level > 0 else 0.4
                colors.append((0.1, green_shade, 0.1))
                points.extend([old_pos, new_pos])
                pos = new_pos
            elif char in self.moving_chars:
                pos = (pos[0]+self.step_length*heading[0], pos[1]+self.step_length*heading[1], 
                       pos[2]+self.step_length*heading[2])
            elif char == "+":
                cos_a, sin_a = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
                heading = (heading[0]*cos_a+left[0]*sin_a, heading[1]*cos_a+left[1]*sin_a, 
                           heading[2]*cos_a+left[2]*sin_a)
                heading = self._normalize(heading)
                left = self._normalize(self._cross_product(up, heading))
            elif char == "-":
                cos_a, sin_a = math.cos(math.radians(-self.angle)), math.sin(math.radians(-self.angle))
                heading = (heading[0]*cos_a+left[0]*sin_a, heading[1]*cos_a+left[1]*sin_a, 
                           heading[2]*cos_a+left[2]*sin_a)
                heading = self._normalize(heading)
                left = self._normalize(self._cross_product(up, heading))
            elif char == "&":
                cos_a, sin_a = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
                heading = (heading[0]*cos_a+up[0]*sin_a, heading[1]*cos_a+up[1]*sin_a, 
                           heading[2]*cos_a+up[2]*sin_a)
                heading = self._normalize(heading)
                up = self._normalize(self._cross_product(heading, left))
            elif char == "^":
                cos_a, sin_a = math.cos(math.radians(-self.angle)), math.sin(math.radians(-self.angle))
                heading = (heading[0]*cos_a+up[0]*sin_a, heading[1]*cos_a+up[1]*sin_a, 
                           heading[2]*cos_a+up[2]*sin_a)
                heading = self._normalize(heading)
                up = self._normalize(self._cross_product(heading, left))
            elif char == "\\":
                cos_a, sin_a = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
                left = (left[0]*cos_a+up[0]*sin_a, left[1]*cos_a+up[1]*sin_a, left[2]*cos_a+up[2]*sin_a)
                left = self._normalize(left)
                up = self._normalize(self._cross_product(heading, left))
            elif char == "/":
                cos_a, sin_a = math.cos(math.radians(-self.angle)), math.sin(math.radians(-self.angle))
                left = (left[0]*cos_a+up[0]*sin_a, left[1]*cos_a+up[1]*sin_a, left[2]*cos_a+up[2]*sin_a)
                left = self._normalize(left)
                up = self._normalize(self._cross_product(heading, left))
            elif char == "|": heading, left = (-heading[0], -heading[1], -heading[2]), (-left[0], -left[1], -left[2])
            elif char == "[":
                stack.append((pos, heading, left, up, branch_level))
                branch_level += 1
                max_branch_level = max(max_branch_level, branch_level)
            elif char == "]" and stack: pos, heading, left, up, branch_level = stack.pop()
        return points, segments, colors
    
    def draw_iteration(self, iteration, ax=None, view_angle=0):
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        word = self.get_word_at_iteration(iteration)
        if word is None:
            print(f"Iterace {iteration} není k dispozici.")
            return None
            
        points, segments, colors = self._process_turtle_commands(word)
        
        if points:
            x_coords, y_coords, z_coords = [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            z_min, z_max = min(z_coords), max(z_coords)
            
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            center_z = (z_min + z_max) / 2
            max_range = max(x_max-x_min, y_max-y_min, z_max-z_min) * 0.55
            
            ax.set_xlim(center_x-max_range, center_x+max_range)
            ax.set_ylim(center_y-max_range, center_y+max_range)
            ax.set_zlim(center_z-max_range, center_z+max_range)
            ax.view_init(elev=20, azim=view_angle)
            
            for i, (start, end) in enumerate(segments):
                color_idx = min(i, len(colors) - 1)
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                        color=colors[color_idx], linewidth=1)
            
            ax.set_title(f"3D fraktální rostlina - Iterace {iteration}")
            ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
            ax.grid(False)
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
            ax.set_facecolor('white')
        return points, segments, colors
    
    def save_last_iteration_png(self):
        last_iteration = len(self.words) - 1
        fig = plt.figure(figsize=(16, 12))
        
        ax1 = fig.add_subplot(221, projection='3d')
        self.draw_iteration(last_iteration, ax1, view_angle=0)
        ax1.set_title("Pohled zepředu")
        
        ax2 = fig.add_subplot(222, projection='3d')
        self.draw_iteration(last_iteration, ax2, view_angle=90)
        ax2.set_title("Pohled z boku")
        
        ax3 = fig.add_subplot(223, projection='3d')
        self.draw_iteration(last_iteration, ax3, view_angle=180)
        ax3.set_title("Pohled zezadu")
        
        ax4 = fig.add_subplot(224, projection='3d')
        self.draw_iteration(last_iteration, ax4, view_angle=45)
        ax4.set_title("Izometrický pohled")
        
        plt.tight_layout()
        png_filename = os.path.splitext(self.filename)[0] + ".png"
        plt.savefig(png_filename, format='png', bbox_inches='tight', dpi=800)
        plt.close(fig)
        print(f"Poslední iterace uložena jako: {png_filename}")
        return png_filename
    
    def create_gif(self):
        start_time = time.time()
        print(f"Generování GIF pro {self.filename}...")
        
        rotation_step = 360 / 36
        
        for iteration in range(len(self.words)):
            iteration_frames = []
            for angle in tqdm(range(0, 360, int(rotation_step))):
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                self.draw_iteration(iteration, ax, view_angle=angle)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                iteration_frames.append(Image.open(buf).copy())
                plt.close(fig)
            self.frames.extend(iteration_frames)
        
        if self.frames:
            self.frames[0].save(self.filename, format='GIF', append_images=self.frames[1:],
                save_all=True, duration=100, loop=0)
            self.save_last_iteration_png()
            duration = time.time() - start_time
            print(f"GIF uložen jako: {self.filename} (doba generování: {duration:.2f}s)")
        else: print(f"Nepodařilo se vytvořit žádné snímky pro GIF.")
        return self.filename

def example_koch_snowflake():
    axiom, rules = "F--F--F", {"F": "F+F--F+F"}
    return LSystem2D(axiom=axiom, rules=rules, angle=60, start_pos=(0, 0), start_angle=0,
                     step_length=1, iterations=4, filename="koch_snowflake.gif").create_gif()

def example_fractal_plant():
    axiom, rules = "X", {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"}
    return LSystem2D(axiom=axiom, rules=rules, angle=25, start_pos=(0, 0), start_angle=35,
                     step_length=1, iterations=8, filename="fractal_plant.gif").create_gif()

def example_dragon_curve():
    axiom, rules = "FX", {"X": "X+YF+", "Y": "-FX-Y"}
    return LSystem2D(axiom=axiom, rules=rules, angle=90, start_pos=(0, 0), start_angle=0,
                     step_length=1, iterations=16, filename="dragon_curve.gif").create_gif()

def example_3d_fractal_plant():
    axiom, rules = "X", {"X": "F-[&[X]+X]+F^[+FX]-X", "F": "FF-[/F^F\\F]+[\\F&F/F]"}
    return LSystem3D(axiom=axiom, rules=rules, angle=22, start_pos=(0, 0, 0), start_heading=(0, 0, 1),
                     start_left=(1, 0, 0), step_length=0.8, iterations=4, filename="3d_fractal_plant.gif").create_gif()

def main():
    output_dir = "output"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    os.chdir(output_dir)
    print("=== Začínám generování L-systémů ===")
    print("\n--- Kochova vločka ---")
    example_koch_snowflake()
    print("\n--- Fraktální rostlina ---")
    example_fractal_plant()
    print("\n--- Dračí křivka ---")
    example_dragon_curve()
    print("\n--- 3D fraktální rostlina ---")
    example_3d_fractal_plant()
    print("\n=== Generování dokončeno ===")

if __name__ == "__main__": main()
