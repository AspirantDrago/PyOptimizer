import random


class BackPropagation:
    def _learn_bprop_sample(self, x, y, is_norm=True, save_sign=False):
        y_out = self.gety(x=x, is_norm=is_norm)
        if is_norm:
            y_norm = [self.normalize(y[i], self.out_norm_to[i], self.out_norm_from[i]) for i in range(len(y))]
            y_out_norm = [self.normalize(y_out[i], self.out_norm_to[i], self.out_norm_from[i]) for i in range(len(y))]
        err = []
        for i in range(min(len(y_norm), len(y_out_norm))):
            err += [(abs(y_norm[i] - y_out_norm[i]) ** self._error_power) / self._error_power]
            if save_sign and y_norm[i] < y_out_norm[i]:
                err[-1] *= -1
            self.output_layer[i].error = err[i]
        return err

    def learn_bprop(self, samples, count_batches=None, is_norm=True, is_shuffle=True):
        if is_shuffle:
            random.shuffle(samples)
        err = [0] * self.count_y
        err_max = [0] * self.count_y
        len_samples = len(samples)
        if not count_batches:
            count_batches = len_samples
        batches = []
        pow_batch = len_samples // count_batches
        for i in range(count_batches - 1):
            batches += [samples[i * pow_batch: (i + 1) * pow_batch]]
        batches += [samples[(count_batches - 1) * pow_batch:]]
        for batch in batches:
            self._prev_learn()
            for x, y in batch:
                arr_err = self._learn_bprop_sample(x, y, is_norm, save_sign=True)
                for i in range(self.count_y):
                    err[i] += abs(arr_err[i])
                    err_max[i] = max(err_max[i], abs(arr_err[i]))
            self._post_learn()
        err = sum(err) / len_samples
        err_max = sum(err_max)
        return err, err_max
