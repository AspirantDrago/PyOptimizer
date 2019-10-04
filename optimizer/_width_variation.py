class _WidthVariation:
    def full_width_variation(self, single_width_variation=None, count=3):
        if not (single_width_variation is None):
            self._single_width_variation = single_width_variation
        if not self.count:
            self.count = count
        self._width_variation = [self._single_width_variation] * self.count
        return self
