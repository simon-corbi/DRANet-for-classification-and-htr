import editdistance
import torch

from src.data.text.best_path_ctc import ctc_best_path_one
from src.data.text.text_to_index import convert_index_to_chars
from src.evaluate.evaluate_recognition import nb_chars_from_list, edit_wer_from_list, nb_words_from_list, ProcessWER
from src.evaluate.metrics_counter import MetricLossCERWER


def evaluate_one_epoch_crnn(data_loader,
                            model,
                            device,
                            char_list,
                            token_blank,
                            ctc_loss_fn,
                            format_wer=ProcessWER.NO,
                            print_frames=False,
                            print_shortcut=True):
    metrics_main = MetricLossCERWER("Main")
    metrics_shortcut = MetricLossCERWER("Shortcut")

    model.eval()

    with torch.no_grad():
        for index_batch, batch_data in enumerate(data_loader):
            x = batch_data["imgs"].to(device)
            x_reduced_len = batch_data["w_reduce"]

            y_enc = batch_data["label_ind"].to(device)
            y_len_enc = batch_data["label_ind_length"]

            y_gt_txt = batch_data["label_str"]

            # Remove text padding
            y_gt_txt = [t.strip() for t in y_gt_txt]
            # Remove double spaces
            y_gt_txt = [t.replace("  ", " ") for t in y_gt_txt]

            nb_item_batch = x.shape[0]

            y_pred, _, _ = model(x)
            output, aux_output = y_pred

            # Main head
            output_log = torch.nn.functional.log_softmax(output, dim=-1)

            ctc_loss = ctc_loss_fn(output_log, y_enc, x_reduced_len, y_len_enc)
            metrics_main.add_loss(ctc_loss.item(), nb_item_batch)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            output_log = output_log.transpose(0, 1)

            top = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in enumerate(output_log)]
            predictions_text = [ctc_best_path_one(p, char_list, token_blank) for p in top]

            predictions_text = [t.strip() for t in predictions_text]  # Remove text padding
            predictions_text = [t.replace("  ", " ") for t in predictions_text]

            cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text)]
            metrics_main.add_cer(sum(cers), nb_chars_from_list(y_gt_txt))

            metrics_main.add_wer(edit_wer_from_list(y_gt_txt, predictions_text, format_wer),
                                 nb_words_from_list(y_gt_txt, format_wer))

            # Shortcut head
            output_log = torch.nn.functional.log_softmax(aux_output, dim=-1)

            ctc_loss = ctc_loss_fn(output_log, y_enc, x_reduced_len, y_len_enc)
            metrics_shortcut.add_loss(ctc_loss.item(), nb_item_batch)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            output_log = output_log.transpose(0, 1)

            top_aux = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                       enumerate(output_log)]

            predictions_text_aux = [ctc_best_path_one(p, char_list, token_blank) for p in top_aux]
            predictions_text_aux = [t.strip() for t in  predictions_text_aux]  # Remove text padding
            predictions_text_aux = [t.replace("  ", " ") for t in predictions_text_aux]

            cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_aux)]
            metrics_shortcut.add_cer(sum(cers), nb_chars_from_list(y_gt_txt))

            metrics_shortcut.add_wer(edit_wer_from_list(y_gt_txt, predictions_text, format_wer),
                                     nb_words_from_list(y_gt_txt, format_wer))

            # # Print first batch prediction
            if index_batch == 0:
                nb_pred_to_print = min(6, nb_item_batch)
                for i in range(nb_pred_to_print):
                    print("-----Ground truth all:-----")
                    print(y_gt_txt[i])
                    print("-----Predictions main:-----")
                    print(predictions_text[i])

                    if print_frames:
                        char_sequence = convert_index_to_chars(top[i], char_list)
                        char_sequence = char_sequence.replace("<BLANK>", "-")
                        print(char_sequence)

                    if print_shortcut:
                        print("-----Predictions shortcut:-----")
                        print(predictions_text_aux[i])
                        if print_frames:
                            char_sequence = convert_index_to_chars(top_aux[i], char_list)
                            char_sequence = char_sequence.replace("<BLANK>", "-")
                            print(char_sequence)

    dict_result = {
        "metrics_main": metrics_main,
        "metrics_shortcut": metrics_shortcut
    }

    return dict_result
