class DictReader:
  def __init__(self, file):
    self.f = open(file, "r")
    self.header = self.f.readline()

    self.data = { }

  def _addElement(self, q, u , a):
    if q not in self.data:
      self.data[q] = { }

    if u not in self.data[q]:
      self.data[q][u] = 0

    self.data[q][u] = max(self.data[q][u], a)

  def load(self):
    for line in self.f:
      question_id, user_id, answered = line.strip('\n').split(',')

      self._addElement(question_id, user_id, answered)

    return self.data

