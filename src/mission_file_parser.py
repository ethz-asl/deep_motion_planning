
class MissionFileParser():

    """Docstring for MissionFileParser. """

    def __init__(self, mission_file):
        self.mission = self.__parse_file__(mission_file)

    def get_mission(self):
        return self.mission

    def __parse_file__(self, mission_file):
        mission = list()

        with open(mission_file, 'r') as file:

            for l in file:
                l = l.strip()
                if l[0] == '#':
                    continue

                parsed = self.__parse_line__(l)
                if not parsed == None:
                    mission.append(parsed)
                else:
                    print('Error while parsing line: {}'.format(l))

        return mission

    def __parse_line__(self, line):
        line_split = line.split(':')

        if len(line_split) == 2:
            if line_split[0] == 'wp':
                return ('wp', self.__parse_waypoint__(line_split[1].strip()))
            elif line_split[0] == 'cmd':
                return ('cmd', self.__parse_command__(line_split[1].strip()))
            else:
                print('Command not supported: {}'.format(line_split[0]))
                return None
        else:
            print('Line has no command')
            return None

    def __parse_waypoint__(self, input):
        waypoint = [float(e) for e in input.split(' ')]
        if len(waypoint) == 3:
            return waypoint
        else:
            print('Waypoint has not 3 elements: {} given'.format(len(waypoint)))
            return None

    def __parse_command__(self, input):
        return input.split('\n')[0]
