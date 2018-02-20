import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


def display_user_participation(pad, save_location):
    """
    Display a pie chart representing the participation according to the length of each authors

    :param save_location: Where to store the figure
    :param pad:
    :return: None
    """
    # Compute author proportions
    authors, proportions = pad.author_proportions(considerate_admin=True)

    # Transform the array in dataframe
    df = pd.DataFrame({'Participation proportion': proportions,
                       'Authors': authors
                       })
    df.index = df['Authors']

    # Plot the results as a pie chart
    plt.figure(figsize=(16, 16))
    df.plot.pie(y='Participation proportion', autopct='%1.0f%%')
    plt.title('Proportion of participation by authors')
    if not os.path.isdir(save_location + '/' + pad.pad_name):
        os.makedirs(save_location + '/' + pad.pad_name)
    plt.savefig(save_location + '/%s/%s_user_participation.png' % (pad.pad_name, pad.pad_name), bbox_inches='tight')


def display_overall_op_type(pad, save_location, jump=False):
    """
    Plot the type counts of all operations of the pad.

    :param save_location:
    :param pad:
    :return: None
    """
    # Initialize the bins and fill them
    types = ['write', 'edit', 'delete', 'paste']
    type_counts = np.zeros(len(types))
    for op in pad.operations:
        if jump or op.type != 'jump':
            type_counts[types.index(op.type)] += 1

    # Transform the array in DataFrame
    df = pd.DataFrame({'Type Counts': type_counts,
                       'Types': types
                       })
    df.index = df['Types']

    # Plot the results as a Seaborn barplot
    plt.figure(figsize=(16, 16))
    sns.barplot(x='Types', y='Type Counts', data=df)
    plt.title('Type repartition of operations of the pad')
    if not os.path.isdir(save_location + '/' + pad.pad_name):
        os.makedirs(save_location + '/' + pad.pad_name)
    plt.savefig(save_location + '/%s/%s_overall_op_type.png' % (pad.pad_name, pad.pad_name), bbox_inches='tight')


def display_types_per_user(pad, save_location, jump=False):
    """
    Plot the type counts of all operations of the pad for all users separately.

    :param jump:
    :param save_location:
    :param pad:
    :return: None
    """

    # Create DataFrame and fill it
    df = pd.DataFrame(columns=('Operations', 'Types', 'Authors'))
    for i, op in enumerate(pad.operations):
        df.loc[i] = [op, op.type, op.author]
    if not jump:
        df = df[df['Types'] != 'jump']

    # Plot the results as a seaborn countplot
    plt.figure(figsize=(16, 16))
    sns.countplot(x='Types', hue="Authors", data=df)
    plt.title('Type repartition of operations of the pad per user', fontsize=25)
    if not os.path.isdir(save_location + '/' + pad.pad_name):
        os.makedirs(save_location + '/' + pad.pad_name)
    plt.tick_params(axis='both', labelsize=20)
    plt.savefig(save_location + '/%s/%s_types_per_user.png' % (pad.pad_name, pad.pad_name), bbox_inches='tight')

def display_proportion_sync_in_pad(pad, save_location):
    """
    Display in a pie chart the proportion of synchronous writing in the entire pad

    :param save_location:
    :param pad:
    :return: None
    """
    prop_sync, prop_async = pad.sync_score()

    df = pd.DataFrame({'sync/async proportion': [prop_sync, prop_async]}, index=['synchronous', 'asynchronous'])

    # Plot the results as a pie chart
    plt.figure(figsize=(16, 16))
    df.plot.pie(y='sync/async proportion', autopct='%1.0f%%', colors=['yellow', 'grey'])
    plt.title('Proportion of {a}synchronous writing in the pad')
    if not os.path.isdir(save_location + '/' + pad.pad_name):
        os.makedirs(save_location + '/' + pad.pad_name)
    plt.savefig(save_location + '/%s/%s_sync_prop_pad.png' % (pad.pad_name, pad.pad_name), bbox_inches='tight')


def display_proportion_sync_in_paragraphs(pad, save_location):
    """
    Display in a barplot the proportion of synchronous writing in all the paragraphs (not the jump lines)

    :param save_location:
    :param pad:
    :return: None
    """
    # Init variables
    prop_sync_paragraphs = []
    paragraph_names = []
    i = 1
    for paragraph in pad.paragraphs:
        if not paragraph.new_line:
            # Create the label of the paragraph
            paragraph_names.append('p' + str(i))
            i += 1
            prop_sync = 0
            for op in paragraph.operations:
                if op.context['synchronous_in_paragraph']:
                    prop_sync += op.context['proportion_paragraph']
            prop_sync_paragraphs.append(prop_sync)
    prop_async_paragraphs = [1 - x for x in prop_sync_paragraphs]

    # Transform the data to a dataframe
    df = pd.DataFrame({'sync proportion': prop_sync_paragraphs, 'async proportion': prop_async_paragraphs},
                      index=paragraph_names)

    # Plot the results as a pie chart
    plt.figure(figsize=(16, 16))
    df.plot.barh(y=['sync proportion', 'async proportion'], stacked=True, color=['yellow', 'grey'])
    plt.gca().invert_yaxis()
    plt.title('Proportion of {a}synchronous writing in paragraphs')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    if not os.path.isdir(save_location + '/' + pad.pad_name):
        os.makedirs(save_location + '/' + pad.pad_name)
    plt.tick_params(axis='y', labelsize=6)
    plt.savefig(save_location + '/%s/%s_sync_prop_para.png' % (pad.pad_name, pad.pad_name), bbox_inches='tight')


def display_user_participation_paragraphs(pad, save_location):
    """
    Display in a barplot the proportion of author writing in all the paragraphs (not the jump lines)

    :param save_location:
    :param pad:
    :return: None
    """
    # Compute the required proportions
    author_names = pad.authors
    paragraph_names, prop_authors_paragraphs = pad.prop_paragraphs()

    # Transform the final data into a pandas dataframe
    df = pd.DataFrame(prop_authors_paragraphs,
                      index=paragraph_names)

    # Plot the results as a stacked plot bar
    plt.figure(figsize=(16, 16))
    df.plot.barh(y=author_names, stacked=True)
    plt.gca().invert_yaxis()
    plt.title('Proportion of absolute writings (overall additions) per authors in paragraphs')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(author_names))
    if not os.path.isdir(save_location + '/' + pad.pad_name):
        os.makedirs(save_location + '/' + pad.pad_name)
    plt.tick_params(axis='y', labelsize=6)
    plt.savefig(save_location + '/%s/%s_user_abs_participation_para.png' % (pad.pad_name, pad.pad_name),
                bbox_inches='tight')


def display_user_participation_paragraphs_with_del(pad, save_location):
    """
    Display in a barplot the proportion of author writing in all the paragraphs (not the jump lines) with the del
    operations

    :param save_location:
    :param pad:
    :return: None
    """
    # Init variables
    author_names = pad.authors
    prop_authors_paragraphs_add = []
    prop_authors_paragraphs_del = []
    paragraph_names = []
    i = 1
    for paragraph in pad.paragraphs:
        # Initialize a dictionary containing participation proportion for each authors
        prop_authors_add = {author_name: 0 for author_name in author_names}
        prop_authors_del = {author_name: 0 for author_name in author_names}
        # Only take into account the real paragraphs (not the new lines)
        if not paragraph.new_line:
            # Create the label of the paragraph
            paragraph_names.append('p' + str(i))
            i += 1
            for elem_op in paragraph.elem_ops:
                if elem_op.operation_type == 'add':
                    prop_authors_add[elem_op.author] += len(elem_op.text_to_add)  # increment with the corresponding len
                if elem_op.operation_type == 'del':
                    prop_authors_del[elem_op.author] += elem_op.length_to_delete  # increment with the corresponding len
            total_count = sum(prop_authors_add.values()) + sum(prop_authors_del.values())
            if total_count != 0:
                prop_authors_add = {k: v / total_count for k, v in prop_authors_add.items()}
                prop_authors_del = {k: v / total_count for k, v in prop_authors_del.items()}
            prop_authors_paragraphs_add.append(prop_authors_add)
            prop_authors_paragraphs_del.append(prop_authors_del)

    # Transform the final data into a pandas DataFrame
    df_add = pd.DataFrame(prop_authors_paragraphs_add, index=paragraph_names)
    # Rename each column by adding write/del in each col to sort them
    col_add = df_add.columns
    col_add = [col_name + ' write/add' for col_name in col_add]
    df_add.rename(columns=dict(zip(df_add.columns, col_add)), inplace=True)

    df_del = pd.DataFrame(prop_authors_paragraphs_del, index=paragraph_names)
    # Rename each column by adding write/del in each col to sort them
    col_del = df_del.columns
    col_del = [col_name + ' del' for col_name in col_del]
    df_del.rename(columns=dict(zip(df_del.columns, col_del)), inplace=True)

    # Concatenate the resulting two dataframes
    df = pd.concat([df_add, df_del], axis=1)
    # Sort them alphabetically to alternate between add and del
    df = df.sort_index(axis=1)

    # Plot the results as a stacked plot bar
    plt.figure(figsize=(16, 16))
    df.plot.barh(y=df.columns, stacked=True, color=sns.color_palette("Paired", 40))
    plt.gca().invert_yaxis()
    plt.title('Proportion of writings per authors in paragraphs with deletions and additions seperated')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(author_names))
    if not os.path.isdir(save_location + '/' + pad.pad_name):
        os.makedirs(save_location + '/' + pad.pad_name)
    plt.tick_params(axis='y', labelsize=6)
    plt.savefig(save_location + '/%s/%s_user_participation_para.png' % (pad.pad_name, pad.pad_name),
                bbox_inches='tight')
    plt.close('all')


def display_boxplot_split(df, metric, save_location=None):
    """
    Create boxplots on the different point of time according to one metric

    :param df: dataframe with all results of metrics
    :param metric: metric to analyze
    :param: save_location: location saved
    :return:
    """
    plt.subplots(figsize=(15, 6))
    sns.boxplot(x='time', y=metric, data=df, color="steelblue")
    plt.title('Distribution of ' + metric + ' at different point in time of all pads')
    plt.savefig(save_location + '/metrics/boxplot_%s.png' % metric,
                bbox_inches='tight')
    plt.close('all')


def display_barplot_split(df, metric, pad_name, save_location=None):
    """
    Plot the distribution over time of a metric of one pad.

    :param df: dataframe with the results of one pad
    :param metric: string corresponding to one metric column of the dataframe
    :param save_location: saving location
    :return:
    """
    plt.subplots(figsize=(15, 6))
    sns.barplot(x='time', y=metric, data=df, color='darkseagreen')
    plt.title('Distribution of ' + metric + ' at different point in time of the pad '+pad_name)
    plt.savefig(save_location + '/%s/evolution_%s.png' % (pad_name, metric),
                bbox_inches='tight')
    plt.close('all')