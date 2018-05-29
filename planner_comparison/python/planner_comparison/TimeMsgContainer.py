import bisect

class TimeMsgContainer():

  def __init__(self):
    self.times = []
    self.msgs = []

  def __len__(self):
    if (len(self.times) == len(self.msgs)):
      return len(self.times)
    else:
      raise Exception("There's something wrong with the vector lengths.")

  def get_data_for_interval(self, start_time, end_time):
    start_time = max(start_time, self.times[0])
    end_time = min(end_time, self.times[-1])
    start_idx = bisect.bisect(self.times, start_time)
    end_idx = bisect.bisect(self.times, end_time)
    time_msg_out = TimeMsgContainer()
    time_msg_out.times = self.times[start_idx:end_idx]
    time_msg_out.msgs = self.msgs[start_idx:end_idx]
    return time_msg_out

  def get_next_msg(self, query_time):
    return self.msgs[bisect.bisect(self.times, query_time)]

  def get_previous_msg(self, query_time):
    return self.msgs[bisect.bisect(self.times, query_time) - 1]