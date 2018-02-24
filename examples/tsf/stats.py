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


class TSFClassifierStats(Stats):
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

class TSFClassifierLMStats(TSFClassifierStats):
  def __init__(self):
    self.reset()

  def reset(self):
    self._loss, self._g, self._ppl, self._lmf, \
      self._df, self._dr, self._ds, \
      self._af, self._ar, self._as, \
      self._lm, self._p0, self._p1 \
      = [], [], [], [], \
      [], [], [], \
      [], [], [], \
      [], [], []
    self._w_loss, self._w_g, self._w_ppl, self._w_lmf,\
      self._w_df, self._w_dr, self._w_ds, \
      self._w_af, self._w_ar, self._w_as, \
      self._w_lm, self._w_p0, self._w_p1 \
      = 0, 0, 0, 0, \
        0, 0, 0, \
        0, 0, 0, \
        0, 0, 0

  def append(self, loss, g, ppl, lmf,
             df, dr, ds,
             af, ar, as_,
             lm, p0, p1,
             w_loss=1., w_g=1., w_ppl=1., w_lmf=1.,
             w_df=1, w_dr=1., w_ds=1,
             w_af=1., w_ar=1., w_as=1.,
             w_lm=1., w_p0=1., w_p1=1.):
    self._loss.append(loss*w_loss)
    self._g.append(g*w_g)
    self._ppl.append(ppl*w_ppl)
    self._lmf.append(lmf*w_lmf)
    self._df.append(df*w_df)
    self._dr.append(dr*w_dr)
    self._ds.append(ds*w_ds)
    self._af.append(af*w_af)
    self._ar.append(ar*w_ar)
    self._as.append(as_*w_as)
    self._lm.append(lm*w_lm)
    self._p0.append(p0*w_p0)
    self._p1.append(p1*w_p1)
    self._w_loss += w_loss
    self._w_g += w_g
    self._w_ppl += w_ppl
    self._w_lmf += w_lmf
    self._w_df += w_df
    self._w_dr += w_dr
    self._w_ds += w_ds
    self._w_af += w_af
    self._w_ar += w_ar
    self._w_as += w_as
    self._w_lm += w_lm
    self._w_p0 += w_p0
    self._w_p1 += w_p1

  @property
  def lmf(self):
    return sum(self._lmf) / self._w_lmf

  @property
  def lm(self):
    return sum(self._lm) / self._w_lm

  @property
  def p0(self):
    return sum(self._p0) / self._w_p0

  @property
  def p1(self):
    return sum(self._p1) / self._w_p1

  def __str__(self):
    return "l %.2f, g %.2f, p %.2f, lmf %.2f, " \
      "df %.2f, dr %.2f, ds %.2f, " \
      "af %.1f, ar %.1f, as %.1f, " \
      "\n lm %.2f, p0 %.2f, p1 %.2f" %(
        self.loss, self.g, self.ppl, self.lmf,
        self.df, self.dr, self.ds,
        self.af*100, self.ar*100, self.as_*100,
        self.lm, self.p0, self.p1)

class TSFClassifierLMRecStats(TSFClassifierLMStats):
  def __init__(self):
    self.reset()

  def reset(self):
    self._loss, self._g, self._ppl, self._lmf, self._rec, \
      self._df, self._dr, self._ds, \
      self._af, self._ar, self._as, \
      self._lm, self._p0, self._p1 \
      = [], [], [], [], [], \
      [], [], [], \
      [], [], [], \
      [], [], []
    self._w_loss, self._w_g, self._w_ppl, self._w_lmf, self._w_rec, \
      self._w_df, self._w_dr, self._w_ds, \
      self._w_af, self._w_ar, self._w_as, \
      self._w_lm, self._w_p0, self._w_p1 \
      = 0, 0, 0, 0, 0, \
        0, 0, 0, \
        0, 0, 0, \
        0, 0, 0

  def append(self, loss, g, ppl, lmf, rec,
             df, dr, ds,
             af, ar, as_,
             lm, p0, p1,
             w_loss=1., w_g=1., w_ppl=1., w_lmf=1., w_rec=1.,
             w_df=1, w_dr=1., w_ds=1,
             w_af=1., w_ar=1., w_as=1.,
             w_lm=1., w_p0=1., w_p1=1.):
    self._loss.append(loss*w_loss)
    self._g.append(g*w_g)
    self._ppl.append(ppl*w_ppl)
    self._lmf.append(lmf*w_lmf)
    self._rec.append(rec*w_rec)
    self._df.append(df*w_df)
    self._dr.append(dr*w_dr)
    self._ds.append(ds*w_ds)
    self._af.append(af*w_af)
    self._ar.append(ar*w_ar)
    self._as.append(as_*w_as)
    self._lm.append(lm*w_lm)
    self._p0.append(p0*w_p0)
    self._p1.append(p1*w_p1)
    self._w_loss += w_loss
    self._w_g += w_g
    self._w_ppl += w_ppl
    self._w_lmf += w_lmf
    self._w_rec += w_rec
    self._w_df += w_df
    self._w_dr += w_dr
    self._w_ds += w_ds
    self._w_af += w_af
    self._w_ar += w_ar
    self._w_as += w_as
    self._w_lm += w_lm
    self._w_p0 += w_p0
    self._w_p1 += w_p1

  @property
  def rec(self):
    return sum(self._rec) / self._w_rec

  def __str__(self):
    return "l %.2f, g %.2f, p %.2f, lmf %.2f, rec %.2f, " \
      "df %.2f, dr %.2f, ds %.2f, " \
      "af %.1f, ar %.1f, as %.1f, " \
      "\n lm %.2f, p0 %.2f, p1 %.2f" %(
        self.loss, self.g, self.ppl, self.lmf, self.rec,
        self.df, self.dr, self.ds,
        self.af*100, self.ar*100, self.as_*100,
        self.lm, self.p0, self.p1)

class TSFClassifierLMRecAdvStats(TSFClassifierLMRecStats):
  def __init__(self):
    self.reset()

  def reset(self):
    self._loss, self._g, self._ppl, self._lmf, self._rec, \
      self._df, self._dr, self._ds, \
      self._af, self._ar, self._as, \
      self._lm, self._p0, self._p1, \
      self._d, self._d0, self._d1 \
      = [], [], [], [], [], \
      [], [], [], \
      [], [], [], \
      [], [], [], \
      [], [], []
    self._w_loss, self._w_g, self._w_ppl, self._w_lmf, self._w_rec, \
      self._w_df, self._w_dr, self._w_ds, \
      self._w_af, self._w_ar, self._w_as, \
      self._w_lm, self._w_p0, self._w_p1, \
      self._w_d, self._w_d0, self._w_d1 \
      = 0, 0, 0, 0, 0, \
        0, 0, 0, \
        0, 0, 0, \
        0, 0, 0, \
        0, 0, 0

  def append(self, loss, g, ppl, lmf, rec,
             df, dr, ds,
             af, ar, as_,
             lm, p0, p1,
             d, d0, d1,
             w_loss=1., w_g=1., w_ppl=1., w_lmf=1., w_rec=1.,
             w_df=1, w_dr=1., w_ds=1,
             w_af=1., w_ar=1., w_as=1.,
             w_lm=1., w_p0=1., w_p1=1.,
             w_d=1, w_d0=1, w_d1=1):
    self._loss.append(loss*w_loss)
    self._g.append(g*w_g)
    self._ppl.append(ppl*w_ppl)
    self._lmf.append(lmf*w_lmf)
    self._rec.append(rec*w_rec)
    self._df.append(df*w_df)
    self._dr.append(dr*w_dr)
    self._ds.append(ds*w_ds)
    self._af.append(af*w_af)
    self._ar.append(ar*w_ar)
    self._as.append(as_*w_as)
    self._lm.append(lm*w_lm)
    self._p0.append(p0*w_p0)
    self._p1.append(p1*w_p1)
    self._d.append(d*w_d)
    self._d0.append(d0*w_d0)
    self._d1.append(d1*w_d1)
    self._w_loss += w_loss
    self._w_g += w_g
    self._w_ppl += w_ppl
    self._w_lmf += w_lmf
    self._w_rec += w_rec
    self._w_df += w_df
    self._w_dr += w_dr
    self._w_ds += w_ds
    self._w_af += w_af
    self._w_ar += w_ar
    self._w_as += w_as
    self._w_lm += w_lm
    self._w_p0 += w_p0
    self._w_p1 += w_p1
    self._w_d += w_d
    self._w_d0 += w_d0
    self._w_d1 += w_d1

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
    return "l %.2f, g %.2f, p %.2f, lmf %.2f, rec %.2f, " \
      "df %.2f, dr %.2f, ds %.2f, " \
      "af %.1f, ar %.1f, as %.1f, " \
      "\n \t\t\t\t lm %.2f, p0 %.2f, p1 %.2f, d %.2f d0 %.2f d1 %.2f" %(
        self.loss, self.g, self.ppl, self.lmf, self.rec,
        self.df, self.dr, self.ds,
        self.af*100, self.ar*100, self.as_*100,
        self.lm, self.p0, self.p1,
        self.d, self.d0, self.d1)
