import time

from analytics import operation_builder
from analytics import parser
import config
from analytics import Operations
from analytics.visualization import *

# path_to_csv = "..\\stian logs\\store.csv"
# list_of_elem_ops_per_pad = get_elem_ops_per_pad_from_ether_csv(path_to_csv)
path_to_db = "etherpad/var/dirty.db"
index_from = 0
dic_author_current_operations_per_pad = dict()
pads = dict()
while True:
    new_list_of_elem_ops_per_pad, index_from = parser.get_elem_ops_per_pad_from_db(path_to_db,
                                                                                   'etherpad',
                                                                                   index_from=index_from)
    if len(new_list_of_elem_ops_per_pad) != 0:
        new_list_of_elem_ops_per_pad_sorted = operation_builder.sort_elem_ops_per_pad(new_list_of_elem_ops_per_pad)
        pads, dic_author_current_operations_per_pad, elem_ops_treated = operation_builder.build_operations_from_elem_ops(
            new_list_of_elem_ops_per_pad_sorted, config.maximum_time_between_elem_ops,
            dic_author_current_operations_per_pad, pads)
        for pad_name in pads:
            pad = pads[pad_name]
            # create the paragraphs
            pad.create_paragraphs_from_ops(elem_ops_treated[pad_name])
            # classify the operations of the pad
            pad.classify_operations(length_edit=config.length_edit, length_delete=config.length_delete)
            # find the context of the operation of the pad
            pad.build_operation_context(config.delay_sync, config.time_to_reset_day, config.time_to_reset_break)

        for pad_name in pads:
            pad = pads[pad_name]
            print("PAD:", pad_name)
            #text = pad.get_text()
            #print(text)
            print('\nCOLORED TEXT BY AUTHOR')
            #pad.display_text_colored_by_authors()

            print('\nCOLORED TEXT BY OPS')
            pad.display_text_colored_by_ops()

    #time.sleep(0.1)