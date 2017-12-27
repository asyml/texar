"""
Class to record training stats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Stats():
  def __init__(self):
    self.reset()

  def reset(self):
    self._loss, self._g, self._ppl, self._d, self._d0, self._d1 \
      = [], [], [], [], [], []
    self._w_loss, self._w_g, self._w_ppl, self._w_d, self._w_d0, self._w_d1 \
      = 0, 0, 0, 0, 0, 0

  def append(self, loss, g, ppl, d, d0, d1,
             w_loss=1., w_g=1., w_ppl=1., w_d=1, w_d0=1., w_d1=1.):
    self._loss.append(loss*w_loss)
    self._g.append(g*w_g)
    self._ppl.append(ppl*w_ppl)
    self._d.append(d*w_d)
    self._d0.append(d0*w_d0)
    self._d1.append(d1*w_d1)
    self._w_loss += w_loss
    self._w_g += w_g
    self._w_ppl += w_ppl
    self._w_d += w_d
    self._w_d0 += w_d0
    self._w_d1 += w_d1

  @property
  def loss(self):
    return sum(self._loss) / self._w_loss

  @property
  def g(self):
    return sum(self._g) / self._w_g

  @property
  def ppl(self):
    return sum(self._ppl) / self._w_ppl

  @property
  def d(self):
    return sum(self._d) / self._w_d

  @property
  def d0(self):
    return sum(self._d0) / self._w_d0

  @property
  def d1(self):
    return sum(self._d1) / self._w_d1

  def __str__(self):
    return "loss %.2f, g %.2f, ppl %.2f d %.2f, adv %.2f %.2f" %(
      self.loss, self.g, self.ppl, self.d, self.d0, self.d1)


class TSFClassifierStats():
  def __init__(self):
    self.reset()

  def reset(self):
    self._loss, self._g, self._ppl, self._df, self._dr, self._ds, \
      self._af, self._ar, self._as = [], [], [], [], [], [], [], [], []
    self._w_loss, self._w_g, self._w_ppl, self._w_df, self._w_dr, self._w_ds, \
      self._w_af, self._w_ar, self._w_as = 0, 0, 0, 0, 0, 0, 0, 0, 0

  def append(self, loss, g, ppl, df, dr, ds, af, ar, as_,
             w_loss=1., w_g=1., w_ppl=1., w_df=1, w_dr=1., w_ds=1,
             w_af=1., w_ar=1., w_as=1.):
    self._loss.append(loss*w_loss)
    self._g.append(g*w_g)
    self._ppl.append(ppl*w_ppl)
    self._df.append(df*w_df)
    self._dr.append(dr*w_dr)
    self._ds.append(ds*w_ds)
    self._af.append(af*w_af)
    self._ar.append(ar*w_ar)
    self._as.append(as_*w_as)
    self._w_loss += w_loss
    self._w_g += w_g
    self._w_ppl += w_ppl
    self._w_df += w_df
    self._w_dr += w_dr
    self._w_ds += w_ds
    self._w_af += w_af
    self._w_ar += w_ar
    self._w_as += w_as

  @property
  def loss(self):
    return sum(self._loss) / self._w_loss

  @property
  def g(self):
    return sum(self._g) / self._w_g

  @property
  def ppl(self):
    return sum(self._ppl) / self._w_ppl

  @property
  def df(self):
    return sum(self._df) / self._w_df

  @property
  def dr(self):
    return sum(self._dr) / self._w_dr

  @property
  def ds(self):
    return sum(self._ds) / self._w_ds

  @property
  def af(self):
    return sum(self._af) / self._w_af

  @property
  def ar(self):
    return sum(self._ar) / self._w_ar

  @property
  def as_(self):
    return sum(self._as) / self._w_as


  def __str__(self):
    return "l %.2f, g %.2f, p %.2f, df %.2f dr %.2f, ds %.2f, "\
      "af %.1f ar %.1f, as %.1f" %(
        self.loss, self.g, self.ppl, self.df, self.dr, self.ds,
        self.af*100, self.ar*100, self.as_*100)
