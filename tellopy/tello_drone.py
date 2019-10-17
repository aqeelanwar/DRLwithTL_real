import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import sys
import time
import contextlib
with contextlib.redirect_stdout(None):
    import pygame

    import pygame.display
    import pygame.key
    import pygame.locals
    import pygame.font
prev_flight_data = None
stat = None
font = None
video_recorder = None


class tello_drone(object):

    class FlightDataDisplay(object):
        # previous flight data value and surface to overlay
        _value = None
        _surface = None
        # function (drone, data) => new value
        # default is lambda drone,data: getattr(data, self._key)
        _update = None
        def __init__(self, key, format, colour=(255,255,255), update=None):
            self._key = key
            self._format = format
            self._colour = colour

            if update:
                self._update = update
            else:
                self._update = lambda drone,data: getattr(data, self._key)

        def update(self, drone, data):
            new_value = self._update(drone, data)
            if self._value != new_value:
                self._value = new_value
                self._surface = font.render(self._format % (new_value,), True, self._colour)
            return self._surface

    def palm_land(self, drone, speed):
        if speed == 0:
            return
        drone.palm_land()

    def toggle_zoom(self, drone, speed):
        # In "video" mode the drone sends 1280x720 frames.
        # In "photo" mode it sends 2592x1936 (952x720) frames.
        # The video will always be centered in the window.
        # In photo mode, if we keep the window at 1280x720 that gives us ~160px on
        # each side for status information, which is ample.
        # Video mode is harder because then we need to abandon the 16:9 display size
        # if we want to put the HUD next to the video.
        if speed == 0:
            return
        drone.set_video_mode(not drone.zoom)
        pygame.display.get_surface().fill((0, 0, 0))

        pygame.display.flip()

    controls = {
        'w': 'forward',
        's': 'backward',
        'a': 'left',
        'd': 'right',
        'space': 'up',
        'left shift': 'down',
        'right shift': 'down',
        'q': 'counter_clockwise',
        'e': 'clockwise',
        # arrow keys for fast turns and altitude adjustments
        'left': lambda drone, speed: drone.counter_clockwise(speed*2),
        'right': lambda drone, speed: drone.clockwise(speed*2),
        'up': lambda drone, speed: drone.up(speed*2),
        'down': lambda drone, speed: drone.down(speed*2),
        'tab': lambda drone, speed: drone.takeoff(),
        'backspace': lambda drone, speed: drone.land(),
        'p': palm_land,
        'z': toggle_zoom
        }

    def flight_data_mode(self, drone, *args):
        return (drone.zoom and "VID" or "PIC")

    def flight_data_recording(self, *args):
        return (video_recorder and "REC 00:00" or "")  # TODO: duration of recording


    # def status_print(text):
    #     pygame.display.set_caption(text)

    hud = [
        FlightDataDisplay('height', 'ALT %3d'),
        FlightDataDisplay('ground_speed', 'SPD %3d'),
        FlightDataDisplay('battery_percentage', 'BAT %3d%%'),
        FlightDataDisplay('wifi_strength', 'NET %3d%%'),
        FlightDataDisplay(None, 'CAM %s', update=flight_data_mode),
        FlightDataDisplay(None, '%s', colour=(255, 0, 0), update=flight_data_recording),
    ]

    # def update_hud(hud, drone, flight_data):
    #     (w,h) = (158,0) # width available on side of screen in 4:3 mode
    #     blits = []
    #     for element in hud:
    #         surface = element.update(drone, flight_data)
    #         if surface is None:
    #             continue
    #         blits += [(surface, (0, h))]
    #         # w = max(w, surface.get_width())
    #         h += surface.get_height()
    #     h += 64  # add some padding
    #     overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    #     overlay.fill((0,0,0)) # remove for mplayer overlay mode
    #     for blit in blits:
    #         overlay.blit(*blit)
    #     pygame.display.get_surface().blit(overlay, (0,0))
    #     pygame.display.update(overlay.get_rect())

    def flightDataHandler(self, event, sender, data):
        global prev_flight_data
        global stat
        stat = data
        text = str(data)
        # if prev_flight_data != text:
        #     update_hud(hud, sender, data)
        #     prev_flight_data = text
    def display(self, frame, manual):
        str_b = "Battery: "+str(stat.battery_percentage)
        text_surface = font.render(str(str_b), True, (219, 53, 53))
        if manual:
            str_m = "Manual"
        else:
            str_m = "Automatic"
        text_surface_manual = font.render(str_m, False, (215, 66, 245))
        # start_time = time.time()
        image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
        # print(image.shape)
        f = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        f = numpy.rot90(f)
        f = pygame.surfarray.make_surface(f)

        pygame.display.flip()
        self.screen.blit(pygame.transform.flip(f, True, False), (0, 0))
        self.screen.blit(text_surface_manual, (840, 0))
        self.screen.blit(text_surface, (0, 0))
        pygame.display.update()

    def get_drone_stats(self):
        drone_stat = {
            'battery_percentage':   stat.battery_percentage,
            'battery_low':          stat.battery_low,
            'camera_state':         stat.camera_state,
            'height':               stat.height,
            'imu_state':            stat.imu_state,
            'wifi_strength':        stat.wifi_strength
        }

        return drone_stat

    def connect_video(self, drone):
        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')
        return container


    def take_action_3(self, drone, action):
        #['Forward', 'Left', 'Right', 'Backward']
        speed = 40
        key_handler = 'custom'

        LY = [0, 0, 0, 0, 0] # no height
        LX = [0, -speed, speed, 0, 0]
        RY = [1*speed, 0, 0, -1*speed, 0]

        getattr(drone, key_handler)(LX[action], LY[action], RY[action])




    # def take_action(self, drone, act, num_act, manual):
    #     speed = 20
    #     key_handler = 'custom'
    #     for e in pygame.event.get():
    #         # WASD for movement
    #         if e.type == pygame.locals.KEYDOWN:
    #             print('+' + pygame.key.name(e.key))
    #             keyname = pygame.key.name(e.key)
    #             if keyname == 'escape':
    #                 drone.quit()
    #                 exit(0)
    #             if keyname == 'm':
    #                 manual = not manual
    #                 getattr(drone, key_handler)(0, 0, 0)
    #
    #     if not manual:
    #         Ry = 2*speed # always go forward
    #         row = int(act / numpy.sqrt(num_act)) - numpy.floor(int(numpy.sqrt(num_act)) / 2)
    #         col = int(act % numpy.sqrt(num_act)) - numpy.floor(int(numpy.sqrt(num_act)) / 2)
    #         print('row:', row)
    #         print('col:', col)
    #         Lx = col*speed
    #         Ly = -row*speed
    #
    #         getattr(drone, key_handler)(Lx, Ly, Ry)
    #
    #     return manual

    def if_takeover(self, drone):
        manual = False
        for e in pygame.event.get():
            # WASD for movement
            if e.type == pygame.locals.KEYDOWN:
                print('+' + pygame.key.name(e.key))
                keyname = pygame.key.name(e.key)
                if keyname == 'escape':
                    drone.quit()
                    exit(0)
                if keyname == 'm':
                    manual = True

        return manual



    def check_action(self, drone, manual, dict):
        # print('Action called')
        speed = 60
        for e in pygame.event.get():
            # WASD for movement
            if e.type == pygame.locals.KEYDOWN:
                print('+' + pygame.key.name(e.key))
                keyname = pygame.key.name(e.key)
                if keyname == 'escape':
                    drone.quit()
                    exit(0)
                if keyname == 'm':
                    manual = not manual

                if keyname == 'l':
                    # save data-tuple
                    numpy.save(dict.stat_path, dict.stat)
                    agent = dict.agent
                    agent.save_network(dict.iteration, dict.network_path)
                    numpy.save(dict.data_path, dict.data_tuple)
                    Mem = dict.Replay_memory
                    Mem.save(load_path=dict.load_path)

                if keyname in self.controls:
                    key_handler = self.controls[keyname]
                    if type(key_handler) == str:
                        getattr(drone, key_handler)(speed)
                    else:
                        key_handler(drone, speed)

            elif e.type == pygame.locals.KEYUP:
                print('-' + pygame.key.name(e.key))
                keyname = pygame.key.name(e.key)
                if keyname in self.controls:
                    key_handler = self.controls[keyname]
                    if type(key_handler) == str:
                        getattr(drone, key_handler)(0)
                    else:
                        key_handler(drone, 0)

        return manual


    def connect(self):
        drone = tellopy.Tello()
        drone.subscribe(drone.EVENT_FLIGHT_DATA, self.flightDataHandler)
        drone.connect()
        drone.wait_for_connection(60.0)
        container = self.connect_video(drone)

        return container, drone

    # def mark_frame_grid(self, action, num_actions,frame):
    #     BLUE = (0, 0, 255)
    #     BLACK = (0, 0, 0)
    #     h = frame.height
    #     w = frame.width
    #     len_a = numpy.round(numpy.sqrt(num_actions))
    #
    #     grid_w = w / len_a
    #     grid_h = h / len_a
    #
    #     a_col = action % len_a
    #     a_row = int(action / len_a)
    #     x = a_col * grid_w
    #     y = a_row * grid_h
    #     width = grid_w
    #     height = grid_h
    #
    #     pygame.draw.rect(self.screen, BLUE, (x, y, width, height), 3)
    #     x = int(x + width / 2)
    #     y = int(y + height / 2)
    #     pygame.draw.circle(self.screen, BLACK, (x, y), 10)
    #     pygame.display.update()

    def mark_frame(self, action, num_actions, frame):
        black_color = (0, 0, 0)
        red_color = (255, 0, 0)
        H = frame.height
        W = frame.width

        pygame.draw.line(self.screen, black_color, (W / 2, H / 2 - H / 4), (W / 2, H / 2 + H / 4), 5)
        pygame.draw.line(self.screen, black_color, (W / 2 - W / 4, H / 2), (W / 2 + W / 4, H / 2), 5)

        y = int(H / 2)

        if action == 0:
            # Forward
            x = int(W / 2)
        elif action == 1:
            # Left
            x = int(W / 2 - H / 6)
        else:
            # Right
            x = int(W / 2 + H / 6)

        pygame.draw.circle(self.screen, red_color, (x, y), 25)
        pygame.display.update()

    def pygame_connect(self, H, W):
        global font
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((H, W))
        pygame.font.init()
        font = pygame.font.SysFont("cmr10", 32)

        return self.screen

    # if __name__ == '__main__':
    #     screen = pygame_connect()
    #     container, drone = connect()
    #
    #     skip_frame = 5
    #     frame_skip = skip_frame
    #
    #     while True:
    #         # flightDataHandler()
    #         for frame in container.decode(video=0):
    #             if 0 < frame_skip:
    #                 frame_skip = frame_skip - 1
    #                 continue
    #                 # print(frame)
    #             else:
    #                 frame_skip = skip_frame
    #                 # Do calculations here
    #                 display(frame, screen)
    #                 check_action(drone)
