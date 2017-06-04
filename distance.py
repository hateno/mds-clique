import math

class Distance():
    def __init__(self, p, q):
        self.p = p
        self.q = q
        if len(p) != len(q):
            raise AssertionError('Vectors p and q must be same length')

    def kl(self):
        d = 0
        for i in range(len(self.p)):
            s = self.p[i] * math.log(self.p[i] / self.q[i])
            d += s
        return d

    def hellinger(self):
        d = 0
        for i in range(len(self.p)):
            s = math.pow(math.sqrt(self.p[i]) - math.sqrt(self.q[i]), 2)
            d += s
        d *= (1 / math.sqrt(2))
        return d

    def tvd(self):
        d = 0
        for i in range(len(self.p)):
            s = abs(self.p[i] - self.q[i])
            d += s
        d /= 2
        return d

    def rd(self, alpha):
        d = 0
        for i in range(len(self.p)):
            s = math.pow(self.p[i], alpha) / math.pow(self.q[i], alpha - 1)
            d += s
        d = (1 / (alpha - 1)) * math.log(d)
        return d
