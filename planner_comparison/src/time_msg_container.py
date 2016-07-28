class TimeMsgContainer():
  
  def __init__(self):
    self.times = []
    self.msgs = []
    
  def __len__(self):
    if (len(self.times) == len(self.msgs)):
      return len(self.times)
    else:
      raise Exception("There's something wrong with the vector lengths.")