from analytics.parser import get_elem_ops_per_pad_from_db
from analytics.operation_builder import sort_elem_ops_per_pad
import config
import run_analytics
import os

list_of_elem_ops_per_pad = dict()
elemOpsCounter = 0
root_of_dbs = "../Data/private/"
# Walk among all the files containing the pads.
for (dirpath, dirnames, filenames) in os.walk(root_of_dbs):
    for filename in filenames:
        if ".sql" in filename:
            path_to_db = os.path.join(dirpath, filename)
            # Fetching the new operations
            list_of_elem_ops_per_main = get_elem_ops_per_pad_from_db(
                path_to_db=path_to_db,
                editor='sql_dump'
            )

            for pad_name, pad_vals in list_of_elem_ops_per_main.items():
                list_of_elem_ops_per_pad[pad_name + filename[-7:-4]] = pad_vals

run_analytics.run(list_of_elem_ops_per_pad,
    texts=True,
    texts_save_location=config.texts_save_location,
    show_visualization=True,
    figs_save_location=config.figs_save_location,
    print_pad_name=True,
    print_text=True,
    print_text_colored_by_authors=True,
    print_text_colored_by_ops=True,
    print_metrics_text=True)
