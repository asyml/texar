"""
Trainer for tsf.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from txt.models.tsf import TSF

from trainer_base import TrainerBase
from utils import log_print, get_batches

class TSFTrainer(TrainerBase):
  """TSF trainer."""
  def __init__(self, hparams=None):
    TrainerBase.__init__(self, hparams)

  @staticmethod
  def default_hparams():
    return {
      "name": "tsf"
      "rho": 1.,
      "gamma_init": 1,
      "gamma_decay": 0.5,
      "gamma_min": 0.001,
      "disp_interval": 1000,
      "batch_size": 128
    }

  def load_data(self):
    hparams = self._hparams
    with open(os.path.join(hparams["data_dir"], "vocab.pkl")) as f:
      vocab = pkl.load(f)
    with open(os.path.join(hparams["data_dir"], "train.pkl")) as f:
      train = pkl.load(f)
    with open(os.path.join(hparams["data_dir"], "val.pkl")) as f:
      val = pkl.load(f)
    with open(os.path.join(hparams["data_dir"], "test.pkl")) as f:
      test = pkl.load(f)

    return vocab, train, val, test

  def eval_model(self, model, sess, vocab, data0, data1, outupt_path):
    batches = utils.get_batches(data0, data1, vocab["word2id"],
                                self._hparams.batch_size, shuffle=False)
    losses = Stats()

    data0_ori, data1_tsf = [], []
    for batch in batches:
      logits_ori, logits_tsf = model.decode_step(batch)

      loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1 = model.eval_step(
        sess, batch, self._hparams.rho, self._hparams.gamma_min)
      batch_size = len(batch["enc_inputs"])
      word_size = batch["weights"].sum()
      losses.append(loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1,
                  w_loss=batch_size, w_g=batch_size,
                  w_ppl=word_size, w_d=batch_size,
                  w_d0=batch_size, w_d1=batch_size)
      ori = utils.logits2word(logits_ori, vocab["word2id"])
      tsf = utils.logits2word(logits_tsf, vocab["word2id"])
      half = self._hparams.batch_size/2
      data0_ori += tsf[:half]
      data1_ori += tsf[half:]

    utils.write_sent(data0_ori, output_path + ".0.tsf")
    utils.write_sent(data1_ori, output_path + ".1.tsf")
    return losses

  def train(self):
    if FLAGS.config:
      with open(FLAGS.config) as f:
        self._hparams = HParams(pkl.load(f))

    log_print("Start training with hparams:")
    log_print(self._hparams)
    if not FLAGS.config:
      with open(os.path.join(self._hparams.expt_dir, self._hparams.name)
                + ".config") as f:
        pkl.dump(self._hparams, f)

    vocab, train, val, test = self.load_data()

    # set vocab size
    self._hparams.vocab_size = vocab["size"]

    # set some hparams here

    with tf.Session() as sess:
      model = TSF(self._hparams)
      log_print("finished building model")

      if FLAGS.model:
        model.saver.restore(ses, FLAGS.model)
      else:
        sess.run(tf.global_variable_initializer())
        sess.run(tf.local_variable_initializer())

      losses = Stats()
      gamma = self._hparams.gamma_init
      step = 0
      for epoch in range(self._hparams["max_epoch"]):
        for batch in utils.get_batches(train[0], train[1], vocab["word2id"],
                                       model._hparams.batch_size, shuffle=True):
          loss_d0 = model.train_d0_step(sess, batch, self._hparams.rho,
                                        self._hparams.gamma,
                                        model._hparams.learning_rate )
          loss_d1 = model.train_d1_step(sess, batch, self._hparams.rho,
                                        self._hparams.gamma,
                                        model._hparams.learning_rate )

          if loss_d0 < 1.2 and loss_d1 < 1.2:
            loss, loss_g, ppl_g, loss_d = model.train_g_step(
              sess, batch, self._hparams.rho, self._hparams.gamma,
              model._hparams.leanring_rate)
          else:
            loss, loss_g, ppl_g, loss_d = model.train_ae_step(
              sess, batch, self._hparams.rho, self._hparams.gamma,
              model._hparams.leanring_rate)

          losses.add(loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1)

          step += 1
          if step % self._hparams.disp_interval:
            log_print(losses)
            losses.reset()

        # eval on dev
        dev_loss = self.eval_model(
          model, sess, vocab, val[0], val[1],
          os.path.join(FLAGS.expt, "sentiment.dev.epoch%d"%(epoch)))
        log_print("dev:" + dev_loss)
        if dev_loss.loss < best_dev:
          best_dev = dev_loss.loss
          file_name = (
            self._hparams['name'] + '_' + '%.2f' %(best_dev) + '.model')
          model.saver.save(
            sess, os.path.join(self._hparams['expt_dir'], file_name),
            latest_filename=hparams['name'] + '_checkpoint',
            global_step=step)
          log_print("saved model %s"%(file_name))

        gamma = max(self._hparams.gamma_min, gamma * self._hparams.gamma_decay)


def main():
  trainer = TSFTrainer()
  trainer.train()

if __name__ == "__main__":
  main()
