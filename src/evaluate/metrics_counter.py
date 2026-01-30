class MetricLossCER(object):
    """
    """
    def __init__(self, tag):
        self.tag = tag
        self.loss = 0
        self.nb_item_loss = 0

        self.cer = 0
        self.nb_letters = 0

    def get_cer(self):
        if self.nb_letters != 0:
            cer_print = self.cer / self.nb_letters
        else:
            cer_print = 1

        return cer_print

    def add_cer(self, cer, nb_letters):
        self.cer += cer
        self.nb_letters += nb_letters

    def add_loss(self, loss, nb_item):
        self.loss += loss
        self.nb_item_loss += nb_item

    def get_loss(self):
        if self.nb_item_loss != 0:
            loss_print = self.loss / self.nb_item_loss
        else:
            loss_print = 1

        return loss_print

    def print_values(self):

        if self.nb_letters != 0:
            cer_print = self.cer / self.nb_letters
        else:
            cer_print = 1

        if self.nb_item_loss != 0:
            loss_print = self.loss / self.nb_item_loss
        else:
            loss_print = 1

        print_str = self.tag
        print_str += " : "
        print_str += f"Loss: {loss_print:.3f}; "
        print_str += f"CER: {100 * cer_print:.2f}%; "

        print(print_str)


class MetricLossCERWER(MetricLossCER):
    """
    """
    def __init__(self, tag):
        super(MetricLossCERWER, self).__init__(tag)

        self.wer = 0
        self.nb_words = 0

    def add_metric(self, other_metrics:MetricLossCER):
        self.wer += other_metrics.wer
        self.nb_words += other_metrics.nb_words

        self.cer += other_metrics.cer
        self.nb_letters += other_metrics.nb_letters

        self.loss += other_metrics.loss
        self.nb_item_loss += other_metrics.nb_item_loss

    def add_wer(self, wer, nb_words):
        self.wer += wer
        self.nb_words += nb_words

    def print_values(self):
        if self.nb_letters != 0:
            cer_print = self.cer / self.nb_letters
        else:
            cer_print = 1.0

        if self.nb_item_loss != 0:
            loss_print = self.loss / self.nb_item_loss
        else:
            loss_print = -1

        if self.nb_words != 0:
            wer_print = self.wer / self.nb_words
        else:
            wer_print = 1.0

        print_str = self.tag
        print_str += " : "
        print_str += f"Loss: {loss_print:.3f}; "
        print_str += f"CER: {100 * cer_print:.2f}%; "
        print_str += f"WER: {100 * wer_print:.2f}%; "

        print(print_str)

    def print_cer(self):

        if self.nb_letters != 0:
            cer_print = self.cer / self.nb_letters

        print_str = self.tag
        print_str += " : "
        print_str += f"CER: {100 * cer_print:.2f}% "

        print(print_str)

    def print_wer(self):

        if self.nb_words != 0:
            wer_print = self.wer / self.nb_words
        else:
            print("No word")
            wer_print = --1

        print_str = self.tag
        print_str += " : "
        print_str += f"WER: {100 * wer_print:.2f}% "

        print(print_str)

    def str_cer_wer(self):
        if self.nb_letters != 0:
            cer_print = self.cer / self.nb_letters
        else:
            cer_print = 1.0

        if self.nb_words != 0:
            wer_print = self.wer / self.nb_words
        else:
            wer_print = 1.0

        print_str = self.tag
        print_str += " : "
        print_str += f"CER: {100 * cer_print:.2f}% "
        print_str += f"WER: {100 * wer_print:.2f}% "

        return print_str

    def print_cer_wer(self):
        print_str = self.str_cer_wer()

        print(print_str)


