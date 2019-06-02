# -*-coding:utf-8-*-
import logging

import torch

# 通过下面的方式进行简单配置输出方式与日志级别
logging.basicConfig(filename='log.txt', level=logging.INFO)


class TrainModel(object):

    def __init__(self, **kwargs):
        super().__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    def _get_inputs(self, feed_dict, use_cuda=True):
        feed_tensor_dict = dict()
        for feature_name in self.feature_names:
            tensor = self.tensor_from_numpy(  # [bs, max_len]
                feed_dict[feature_name], use_cuda=use_cuda)
            feed_tensor_dict[feature_name] = tensor
        return feed_tensor_dict

    def fit(self):
        """训练模型
        """
        best_dev_loss = 1.e8
        current_patience = 0
        for epoch in range(self.nb_epoch):
            train_loss, dev_loss = 0., 0.
            self.encoder.train()
            self.ptr_model.train()
            self.decoder_model.train()
            if self.lr_decay != 0.:
                self.optimizer = self.decay_learning_rate(epoch, self.learning_rate)
            for i, feed_dict in enumerate(self.data_iter_train):
                self.optimizer.zero_grad()
                feed_tensor_dict = self._get_inputs(feed_dict, self.encoder.use_cuda)
                segment = self.tensor_from_numpy(feed_dict["segment"], 'long', self.encoder.use_cuda)
                label = self.stack_label_tensor(feed_dict['label'], 'long', self.encoder.use_cuda)
                # encoder
                if self.encoder.rnn_unit_type == 'lstm':
                    encoder_output, (encoder_hn, _) = self.encoder(**feed_tensor_dict)  # encoder_state: (bs * L, H)
                else:
                    encoder_output, encoder_hn = self.encoder(**feed_tensor_dict)  # encoder_state: (bs * L, H)

                # mask
                mask = feed_tensor_dict[str(self.feature_names[0])] > 0
                # ptr network
                ptr_logits = self.ptr_model(feed_tensor_dict, encoder_output)
                ptr_loss = self.ptr_model.loss(ptr_logits, mask, segment)
                # decoder network
                label_logits = self.decoder_model(feed_tensor_dict, segment, encoder_output)
                label_loss = self.decoder_model.loss(label_logits, label)
                loss = ptr_loss + label_loss
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                print('Epoch {0} / {1}: data {2} / {3}\r'.format(epoch + 1, self.nb_epoch,
                                                                 self.data_iter_train.iter_variable,
                                                                 self.data_iter_train.data_count))
            print(
                'Epoch {0} / {1}: data {2} / {3}\n'.format(epoch + 1, self.nb_epoch, self.data_iter_train.data_count,
                                                           self.data_iter_train.data_count))
            # 计算开发集loss
            self.encoder.eval()
            self.ptr_model.eval()
            self.decoder_model.eval()
            # dev_labels_pred, dev_labels_gold = [], []
            for feed_dict in self.data_iter_dev:
                feed_tensor_dict = self._get_inputs(feed_dict, self.encoder.use_cuda)
                segment = self.tensor_from_numpy(feed_dict["segment"], 'long', self.encoder.use_cuda)
                label = self.stack_label_tensor(feed_dict['label'], 'long', self.encoder.use_cuda)

                if self.encoder.rnn_unit_type == 'lstm':
                    encoder_output, (encoder_hn, _) = self.encoder(**feed_tensor_dict)  # encoder_state: (bs * L, H)
                else:
                    encoder_output, encoder_hn = self.encoder(**feed_tensor_dict)  # encoder_state: (bs * L, H)
                # mask
                mask = feed_tensor_dict[str(self.feature_names[0])] > 0
                # ptr network
                ptr_logits = self.ptr_model(feed_tensor_dict, encoder_output)
                ptr_loss = self.ptr_model.loss(ptr_logits, mask, segment)
                # decoder network
                label_logits = self.decoder_model(feed_tensor_dict, segment, encoder_output)
                label_loss = self.decoder_model.loss(label_logits, label)
                loss = ptr_loss + label_loss

                dev_loss += loss.item()
            logging.info('\ttrain loss: {0}, dev loss: {1}'.format(train_loss, dev_loss))

            # 判断是否需要保存模型
            if dev_loss < best_dev_loss:
                current_patience = 0
                best_dev_loss = dev_loss
                # 保存模型
                self.save_model()
                logging.info('\tmodel has saved to {0}!'.format(self.path_save_model))
            else:
                current_patience += 1
                logging.info('\tno improvement, current patience: {0} / {1}'.format(
                    current_patience, self.max_patience))
                if self.max_patience <= current_patience:
                    logging.info('finished training! (early stopping, max patience: {0})'.format(self.max_patience))
                    return
        logging.info('finished training!')

    def decay_learning_rate(self, epoch, init_lr):
        """衰减学习率

        Args:
            epoch: int, 迭代次数
            init_lr: 初始学习率
        """
        lr = init_lr / (1 + self.lr_decay * epoch)
        logging.info('learning rate: {0}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    @staticmethod
    def tensor_from_numpy(data, dtype='long', use_cuda=True):
        """将numpy转换为tensor
        Args:
            data: numpy
            dtype: long or float
            use_cuda: bool
        """
        assert dtype in ('long', 'float')
        if dtype == 'long':
            data = torch.from_numpy(data).long()
        else:
            data = torch.from_numpy(data).float()
        if use_cuda:
            data = data.cuda()
        return data

    @staticmethod
    def stack_label_tensor(list_data, dtype='long', use_cuda=True):
        """将list转换为tensor
        Args:
            list_data: list
            dtype: long or float
            use_cuda: bool
        """
        import numpy as np
        assert dtype in ('long', 'float')
        data = []
        for l in list_data:
            data.extend(l)
        if dtype == 'long':
            data = torch.from_numpy(np.array(data)).long()
        else:
            data = torch.from_numpy(np.array(data)).float()
        if use_cuda:
            data = data.cuda()
        return data

    def save_model(self):
        """保存模型
        """
        import os
        torch.save(self.encoder.state_dict(), os.path.join(self.path_save_model, "encoder"))
        torch.save(self.ptr_model.state_dict(), os.path.join(self.path_save_model, "pointer_network"))
        torch.save(self.decoder_model.state_dict(), os.path.join(self.path_save_model, "decoder"))

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.data_iter_train.batch_size = batch_size
        self.data_iter_dev.batch_size = batch_size

    def set_max_patience(self, max_patience):
        self.max_patience = max_patience

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate
