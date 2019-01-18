from analytics import operation_builder
from analytics.Operations import ElementaryOperation, Paragraph, SuperParagraph, Operation
import config
from itertools import combinations
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy import optimize
from scipy import spatial
import spacy
import regex as re
from spacy.tokenizer import Tokenizer
import warnings

def get_colors():
    '''
    :return:  color list
    '''
    colors = []
    for i in range(30, 38):
        colors.append('\033[' + str(i) + 'm')
    for i in range(90, 97):
        colors.append('\033[' + str(i) + 'm')
    return colors


Stoplist = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_md')

prefix_re = re.compile(r'''^[\[\("']''')
suffix_re = re.compile(r'''[\]\)"']$''')
infix_re = re.compile(r'''[-~]''')
simple_url_re = re.compile(r'''^https?://''')

def custom_tokenizer(nlp):
    '''
    this is used to improve spacy nlp function to have a better performance on url recognization.
    :param nlp:
    :return:
    '''
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=simple_url_re.match)
def cleanText(text):
    '''
    :param text: the text that needs to be cleaned
    :return:  cleaned text
    '''
    cleaned_text = []
    doc = nlp(text)
    for word in doc:
        if not word.is_punct:
            if (word.text.lower() not in Stoplist and
                not word.like_num and not word.like_url and '\n' not in word.string):
                cleaned_text.append(word.lemma_) ## add the prototype of the word
    return ' '.join(cleaned_text)

class Pad:
    """
    Pad. Contains all the operations for a particular pad.
    """

    def __init__(self, pad_name):
        """
        Create a pad
        :param pad_name:  name of the new pad
        """

        self.authors = []
        """:type: list[str]"""

        self.operations = []
        """:type: list[Operation]"""

        self.pad_name = pad_name
        """:type: str"""

        self.paragraphs = []
        """:type: list[Paragraph]"""

        self.all_paragraphs = [] # All paragraphs that have ever been added
        """:type: list[Paragraph]"""

        self.superparagraphs = []
        """:type: list[SuperParagraph]"""

        self.author_operation= {}
        self.author_timeStamp = {}
        self.author_vectors = {}
        self.author_text = {}
        self.author_distance ={}
        self.author_similarity = {}
        self.start_time = float('inf')
        self.window_operation = {}
        self.distance={}
        self.similarity = {}
        self.window_operation_text={}
        self.fitting_slope = 0


    ###############################
    # Complete content of the Pad
    ###############################
    def add_operation(self, operation):
        """
        Add an Operation to the list of ops
        :param operation: Operation to add
        """
        # Skip if the operation is already added
        if not operation.pushed:
            operation.pushed = True
            self.operations.append(operation)

    def add_operations(self, operations):
        """
        Add a list of Operation to the list of ops
        :param operations: list of Operation to add
        """
        for op in operations:
            self.add_operation(op)

    def get_elem_ops(self, sorted_):
        """
        Get the list of ElementaryOperation from all the Operation.
        The result is ordered by timestamp. Good for building a representation
        of the text. Note that each ElementaryOperation knows the Operation it
        belongs to.
        :param sorted_: boolean saying if the operations should be sorted
        :return: list of ElementaryOperation
        :rtype: list[ElementaryOperation]
        """
        # Recover all the elementary ops
        elem_ops = [elem_op for op in self.operations for elem_op in op.elem_ops]
        if sorted_:
            return ElementaryOperation.sort_elem_ops(elem_ops)
        else:
            return elem_ops

    def get_absolute_index(self, paragraph_index):
        """
        Find the absolute index of the paragraph in position 'paragraph_index'.
        The absolute index is the index within the 'all_paragraphs' array.
        It uses the id of the paragraph, which should not change in time.
        It is useful to keep track of all paragraphs that have been added or
        removed in time.
        Returns -1 if the paragraph is not found in 'all_paragraphs'.
        :param paragraph_index: index of the paragraph in self.paragraphs
        :return: absolute index
        :rtype: int
        """
        paragraph_id = self.paragraphs[paragraph_index].paragraph_id

        for para_idx, para in enumerate(self.all_paragraphs):
            if para.paragraph_id == paragraph_id:
                return para_idx
        else:
            return -1

    def delete_paragraphs(self, idx, abs_idx=None, action="just_delete",
        ignore_assertions=False):
        """
        Delete paragraph in position "idx" from self.paragraphs and set the
        "deleted" flag in self.all_paragraphs.
        It also updates the indices of the superparagraphs in
        self.superparagraphs depending on "action" (reason why the paragraph is
        deleted).
        :param idx: index of the paragraph in self.paragraphs
        :param abs_idx: index of the paragraph in self.all_paragraphs
        :param action: "just_delete" (delete the paragraph)
            or "split" (we want to split current paragraph into three)
        """
        found_superpara = None
        if abs_idx is not None:
            self.all_paragraphs[abs_idx].is_deleted = True

        # If "just_delete", then we deleted the paragraph
        # otherwise, we splitted it into three
        length_change = -1 if action == "just_delete" else 2
        remove_first = False
        remove_last = False

        for sidx, superpara in enumerate(self.superparagraphs):
            if (idx >= superpara.start and
                idx < superpara.start + superpara.length):
                # This is the superpara involved
                if idx == superpara.start:
                    remove_first = True
                elif idx == superpara.start + superpara.length - 1:
                    remove_last = True
                # Modify the length
                superpara.length += length_change
                # Keep track that superpara involved was found
                found_superpara = sidx

            elif (found_superpara is not None and
                idx < superpara.start):
                # If superparas after superpara involved,
                # its start is shifted
                superpara.start += length_change

        if found_superpara is None:
            assert ignore_assertions or False # Shouldn't happen

        # If we are not splitting, we need to update the superparagraphs'
        # positions looking at what we are deleting because they may combine
        # or separate.
        # If we are splitting a paragraph into paragraph+new_line+paragraph,
        # it is already good to go.
        elif action == "just_delete":
            # Delete a new_line paragraph
            if self.paragraphs[idx].new_line:
                # new_line superpara has length 1: combine or modify it
                if (self.superparagraphs[found_superpara].new_line and
                    self.superparagraphs[found_superpara].length == 1):
                    # Has to be combined with previous and next superparas
                    if (found_superpara > 0 and
                        found_superpara < len(self.superparagraphs) - 1):
                        # The superpara next and previous to this should not be
                        # new_line (otherwise they would have been merged)
                        assert ignore_assertions or not self.superparagraphs[found_superpara - 1].new_line
                        assert ignore_assertions or not self.superparagraphs[found_superpara + 1].new_line
                        # Generate id for the combined superpara
                        superpara_id = Paragraph.compute_para_id(
                            "merge",
                            self.superparagraphs[found_superpara - 1].id,
                            self.superparagraphs[found_superpara + 1].id)
                        # Combine with the previous and next superparas
                        self.superparagraphs[found_superpara - 1].id = superpara_id
                        self.superparagraphs[found_superpara - 1].length += (
                            self.superparagraphs[found_superpara].length +
                            self.superparagraphs[found_superpara + 1].length)
                        # Delete the extra superparas
                        del self.superparagraphs[found_superpara]
                        del self.superparagraphs[found_superpara]

                    # Has to be combined with previous superpara
                    elif found_superpara > 0:
                        # The superpara previous to this should not be a new_line
                        # (otherwise they would have been merged previously)
                        assert ignore_assertions or not self.superparagraphs[found_superpara - 1].new_line
                        # Combine with previous superpara
                        self.superparagraphs[found_superpara - 1].length += (
                            self.superparagraphs[found_superpara].length)
                        # Delete it
                        del self.superparagraphs[found_superpara]

                    # Has to be combined with next superpara
                    elif found_superpara < len(self.superparagraphs) - 1:
                        # The superpara next to this should not be a new_line
                        # (otherwise they would have been merged previously)
                        assert ignore_assertions or not self.superparagraphs[found_superpara + 1].new_line
                        # Combine with next superpara
                        self.superparagraphs[found_superpara + 1].length += (
                            self.superparagraphs[found_superpara].length)
                        self.superparagraphs[found_superpara + 1].start = (
                            self.superparagraphs[found_superpara].start)
                        # Delete it
                        del self.superparagraphs[found_superpara]

                    # Doesn't need to be combined (just change its type)
                    else:
                        # Turn into non-newline superpara
                        self.superparagraphs[found_superpara].new_line = False

                # found_superpara's position within self.superparagraphs is
                # valid and its length is 0: delete it
                if (found_superpara < len(self.superparagraphs) and
                    self.superparagraphs[found_superpara].length == 0):
                    # Should not be a new_line
                    assert ignore_assertions or not self.superparagraphs[found_superpara].new_line
                    # Just delete it
                    del self.superparagraphs[found_superpara]

            # Delete a non-new_line paragraph (from a non-new_line superpara)
            else:
                # Right now we have not combined
                combined = False
                # Superpara has length 0
                if (self.superparagraphs[found_superpara].length == 0):
                    # Superpara is neither first nor last
                    if (found_superpara > 0 and
                        found_superpara < len(self.superparagraphs) - 1):
                        # Delete text superpara between two new_line superparas
                        assert ignore_assertions or self.superparagraphs[found_superpara - 1].new_line
                        assert ignore_assertions or self.paragraphs[idx - 1].new_line
                        # Combine previous and next new_line superparas
                        self.superparagraphs[found_superpara - 1].length += (
                            self.superparagraphs[found_superpara + 1].length)
                        del self.superparagraphs[found_superpara + 1]
                    # Delete empty superpara and set "combined" flag
                    del self.superparagraphs[found_superpara]
                    combined = True

                # Superpara has length 1
                elif (self.superparagraphs[found_superpara].length == 1):
                    # Removed the first paragraph of the superpara
                    if remove_first:
                        # Involved superparagraph is not the first and
                        # it contains one new_line paragraph
                        if (found_superpara > 0 and
                            self.paragraphs[idx + 1].new_line):
                            # Prev superpara should be a new_line
                            assert ignore_assertions or self.superparagraphs[found_superpara - 1].new_line
                            # Combine current superpara with prev superpara
                            self.superparagraphs[found_superpara - 1].length += 1
                            del self.superparagraphs[found_superpara]
                            combined = True
                        # It's the first superparagraph
                        elif (found_superpara == 0 and
                            self.paragraphs[idx + 1].new_line):
                            # Remove its id
                            self.superparagraphs[found_superpara].id = (
                                Paragraph.compute_para_id('?'))

                    # Removed the last paragraph of the superpara
                    else:
                        assert ignore_assertions or remove_last
                        # The previous paragraph should be a new_line
                        assert ignore_assertions or self.paragraphs[idx - 1].new_line
                        # Not the last superpara
                        if found_superpara < len(self.superparagraphs) - 1:
                            # Next superpara is new_line
                            assert ignore_assertions or self.superparagraphs[found_superpara + 1].new_line
                            # Combine with the next new_line superpara
                            self.superparagraphs[found_superpara + 1].length += 1
                            self.superparagraphs[found_superpara + 1].start -= 1
                            del self.superparagraphs[found_superpara]
                            combined = True

                # Superpara has length 2
                elif (self.superparagraphs[found_superpara].length == 2):
                    # Safety checks
                    if remove_first and self.paragraphs[idx + 1].new_line:
                        assert ignore_assertions or not self.paragraphs[idx + 2].new_line
                    elif remove_last:
                        assert ignore_assertions or not self.paragraphs[idx - 2].new_line
                    else:
                        assert ignore_assertions or self.paragraphs[idx - 1].new_line
                        assert ignore_assertions or self.paragraphs[idx - 1].new_line

                    # We removed the text para in a superpara of the form:
                    # (new_line, text, new_line)
                    # We need to turn it into a new_line superpara
                    if (not (remove_first or remove_last) and
                        self.paragraphs[idx - 1].new_line and
                        self.paragraphs[idx + 1].new_line):
                        self.superparagraphs[found_superpara].new_line = True

                # We still have to check if we need to combine superparagraphs
                if not combined and (self.superparagraphs[found_superpara].length > 2 or
                    (self.superparagraphs[found_superpara].length == 2 and
                        (remove_first or remove_last))):
                    # Remove first para of the superpara
                    if remove_first:
                        # Check if need to combine with prev newlines
                        if found_superpara > 0 and self.paragraphs[idx + 1].new_line:
                            assert ignore_assertions or self.superparagraphs[found_superpara - 1].new_line
                            # Combine newline with previous superpara
                            self.superparagraphs[found_superpara - 1].length += 1
                            self.superparagraphs[found_superpara].length -= 1
                            self.superparagraphs[found_superpara].start += 1

                    # Remove last para of the superpara
                    elif remove_last:
                        # Check if need to combine with next newlines
                        if found_superpara < len(self.superparagraphs) - 1:
                            assert ignore_assertions or self.superparagraphs[found_superpara + 1].new_line
                            # Combine with next superpara
                            self.superparagraphs[found_superpara + 1].length += 1
                            self.superparagraphs[found_superpara + 1].start -= 1
                            self.superparagraphs[found_superpara].length -= 1

                    # Remove a paragraph in the middle of the superpara
                    # that was surrounded by new_line paras
                    elif (self.paragraphs[idx - 1].new_line and
                        self.paragraphs[idx + 1].new_line):
                        # We need to split the superpara in two:

                        # Compute lengths of the new superparas
                        sp_start1 = self.superparagraphs[found_superpara].start
                        sp_length1 = (idx - 1 -
                            self.superparagraphs[found_superpara].start)
                        sp_start2 = sp_start1 + sp_length1 + 2
                        sp_length2 = (- sp_length1 - 2 +
                            self.superparagraphs[found_superpara].length)
                        # Get ids for new superparas
                        new_ids = Paragraph.compute_para_id("split",
                            self.superparagraphs[found_superpara].id)

                        # Splitting results in two valid superparas
                        if sp_length1 > 0 and sp_length2 > 0:
                            self.superparagraphs[found_superpara].start = sp_start2
                            self.superparagraphs[found_superpara].length = sp_length2
                            self.superparagraphs[found_superpara].id = new_ids[1]

                            new_super_para = SuperParagraph(
                                sp_start1, sp_length1, False, new_ids[0])
                            new_super_para_nl = SuperParagraph(
                                sp_start1 + sp_length1, 2, True, 'x')

                            self.superparagraphs.insert(found_superpara,
                                new_super_para_nl)
                            self.superparagraphs.insert(found_superpara,
                                new_super_para)

                        # Only first superpara is valid
                        elif sp_length1 > 0:
                            # -2 because we removed one text para and
                            # its previous new_line para
                            assert ignore_assertions or (sp_length1 ==
                                self.superparagraphs[found_superpara].length - 2)
                            self.superparagraphs[found_superpara].length = sp_length1
                            new_super_para_nl = SuperParagraph(
                                sp_start1 + sp_length1, 2, True, 'x')
                            self.superparagraphs.insert(found_superpara + 1,
                                new_super_para_nl)

                        # Only second superpara is valid
                        elif sp_length2 > 0:
                            assert ignore_assertions or (sp_length2 ==
                                self.superparagraphs[found_superpara].length - 2)
                            assert ignore_assertions or (sp_start2 ==
                                self.superparagraphs[found_superpara].start + 2)
                            self.superparagraphs[found_superpara].start = sp_start2
                            self.superparagraphs[found_superpara].length = sp_length2
                            new_super_para_nl = SuperParagraph(
                                sp_start1 + sp_length1, 2, True, 'x')
                            self.superparagraphs.insert(found_superpara,
                                new_super_para_nl)

                        else:
                            assert ignore_assertions or False # Should not happen
                    else:
                        # Special case when removing many paras one
                        # after the other. We take care of combining
                        # after the last one is removed.
                        pass

        # Lastly, remove from self.paragraphs
        del self.paragraphs[idx]

    def insert_paragraphs(self, idx, para, ignore_assertions=False):
        """
        Insert paragraph "para" in position "idx" of self.paragraphs
        and in the corresponding index of self.all_paragraphs.
        Also update indices of superparagraphs in self.superparagraphs.
        :param idx: index of the paragraph in self.paragraphs
        :param para: paragraph to be inserted.
        """
        # No superparagraphs created: Create the initial superpara
        if len(self.superparagraphs) == 0:
            new_super_para = SuperParagraph(0, 1, False,
                Paragraph.compute_para_id("initial"))
            self.superparagraphs.append(new_super_para)

        # There are existing superparagraphs
        else:
            # Find affected superpara and update start indices of next ones
            found_superpara = None
            for sidx, superpara in enumerate(self.superparagraphs):
                if (found_superpara is not None):
                    # Next superparas need to start after
                    superpara.start += 1
                elif (idx > superpara.start and
                    idx < superpara.start + superpara.length):
                    # Otherwise it'd have chosen the previous superpara
                    if not (para.new_line or superpara.new_line):
                        assert ignore_assertions or para.new_line or superpara.new_line
                    superpara.length += 1
                    found_superpara = sidx
                elif (superpara.new_line and para.new_line and
                    (idx == superpara.start or
                        idx == superpara.start + superpara.length)):
                    # This is the superpara involved
                    superpara.length += 1
                    found_superpara = sidx

            # We didn't find the relevant superpara's position:
            # Handle all possible cases for this situation
            if found_superpara is None:
                if idx == 0:
                    if len(self.superparagraphs) >= 2:
                        for superpara in self.superparagraphs[1:]:
                            superpara.start += 1

                    if para.new_line and not self.superparagraphs[0].new_line:
                        if self.paragraphs[0].new_line:
                            self.superparagraphs[0].start += 2
                            self.superparagraphs[0].length -= 1
                            new_super_para = SuperParagraph(0, 2, True, 'x')
                            self.superparagraphs.insert(0, new_super_para)
                            if self.superparagraphs[1].length == 0:
                                del self.superparagraphs[1]
                        else:
                            self.superparagraphs[0].length += 1

                    elif (not para.new_line and
                        self.superparagraphs[0].new_line):
                        self.superparagraphs[0].start += 1
                        new_super_para = SuperParagraph(0, 1, False,
                            Paragraph.compute_para_id("?"))
                        self.superparagraphs.insert(0, new_super_para)

                    elif (not para.new_line and
                        not self.superparagraphs[0].new_line):
                        assert ignore_assertions or self.paragraphs[0].new_line
                        self.superparagraphs[0].length += 1

                    else:
                        assert ignore_assertions or False # Shouldnt happen

                elif idx == len(self.paragraphs):
                    if para.new_line and not self.superparagraphs[-1].new_line:

                        if self.paragraphs[-1].new_line:
                            self.superparagraphs[-1].length -= 1
                            new_super_para = SuperParagraph(
                                self.superparagraphs[-1].start + self.superparagraphs[-1].length,
                                2, True, 'x')
                            self.superparagraphs.append(new_super_para)
                        else:
                            self.superparagraphs[-1].length += 1
                    elif not para.new_line and self.superparagraphs[-1].new_line:
                        new_super_para = SuperParagraph(
                            self.superparagraphs[-1].start + self.superparagraphs[-1].length,
                            1, False, Paragraph.compute_para_id("?"))
                        self.superparagraphs.append(new_super_para)
                    elif not para.new_line and not self.superparagraphs[-1].new_line:
                        assert ignore_assertions or self.paragraphs[-1].new_line, self.pad_name
                        self.superparagraphs[-1].length += 1
                    else:
                        assert ignore_assertions or False # Shouldnt happen

                else:
                    assert ignore_assertions or False # Shouldnt happen

            # We found the relevant superpara's position
            else:
                # Inserting text inside new_line superpara
                if (self.superparagraphs[found_superpara].new_line and
                    not para.new_line):
                    # We need to split the new_line superpara in two and
                    # create a text superpara for the text para
                    first1 = self.superparagraphs[found_superpara].start
                    last1 = idx - 1
                    len1 = last1 - first1 + 1
                    first2 = idx
                    len2 = self.superparagraphs[found_superpara].length - len1 - 1
                    last2 = first2 + len2 - 1
                    if len1 >= 2 and len2 >= 2:
                        # Superpara becomes 3 superparas
                        new_super_para = SuperParagraph(
                            idx, 1, False, Paragraph.compute_para_id("?"))
                        new_super_para_nl = SuperParagraph(
                            self.superparagraphs[found_superpara].start,
                            len1, True, 'x')
                        self.superparagraphs[found_superpara].start = idx + 1
                        self.superparagraphs[found_superpara].length = len2
                        self.superparagraphs.insert(found_superpara, new_super_para)
                        self.superparagraphs.insert(found_superpara, new_super_para_nl)

                    elif len1 >= 2:
                        # Superpara becomes 2 superparas
                        if (len(self.superparagraphs) - 1 > found_superpara and
                            not self.superparagraphs[found_superpara + 1].new_line):
                            assert ignore_assertions or (not
                                self.paragraphs[
                                    self.superparagraphs[
                                        found_superpara + 1].start - 1].new_line)
                            # We merge with superpara afterwards
                            self.superparagraphs[found_superpara + 1].start = idx
                            self.superparagraphs[found_superpara + 1].length += len2 + 1
                            self.superparagraphs[found_superpara].length = len1

                        else:
                            # We dont merge with superpara afterwards
                            new_super_para = SuperParagraph(
                                idx, len2 + 1, False,
                                Paragraph.compute_para_id("?"))
                            self.superparagraphs[found_superpara].length = len1
                            self.superparagraphs.insert(found_superpara + 1,
                                new_super_para)

                    elif len2 >= 2:
                        # Superpara becomes 2 superparas
                        if (found_superpara > 0 and
                            not self.superparagraphs[found_superpara - 1].new_line):
                            # We merge with previous superpara
                            self.superparagraphs[found_superpara - 1].length += len1 + 1
                            self.superparagraphs[found_superpara].length = len2
                            self.superparagraphs[found_superpara].start = idx + 1
                        else:
                            # We dont merge with previous superpara
                            new_super_para = SuperParagraph(
                                self.superparagraphs[found_superpara].start,
                                len1 + 1, False, Paragraph.compute_para_id("?"))
                            self.superparagraphs[found_superpara].length = len2
                            self.superparagraphs[found_superpara].start = idx + 1
                            self.superparagraphs.insert(found_superpara,
                                new_super_para)

                    else:
                        # Superpara turns into a non-newline
                        if (len(self.superparagraphs) - 1 > found_superpara and
                            not self.superparagraphs[found_superpara + 1].new_line and
                            found_superpara > 0 and
                            not self.superparagraphs[found_superpara - 1].new_line):
                            # We merge with superparas afterwards and previous
                            self.superparagraphs[found_superpara - 1].length += (1 +
                                len1 + len2 +
                                self.superparagraphs[found_superpara + 1].length)
                            new_sid = Paragraph.compute_para_id("merge",
                                reference_id1=self.superparagraphs[found_superpara - 1].id,
                                reference_id2=self.superparagraphs[found_superpara + 1].id)
                            self.superparagraphs[found_superpara - 1].id = new_sid
                            del self.superparagraphs[found_superpara]
                            del self.superparagraphs[found_superpara]

                        elif (len(self.superparagraphs) - 1 > found_superpara and
                            not self.superparagraphs[found_superpara + 1].new_line):
                            assert ignore_assertions or (not
                                self.paragraphs[
                                    self.superparagraphs[
                                        found_superpara + 1].start].new_line)

                            # Merge with superpara afterwards
                            self.superparagraphs[found_superpara + 1].length += (
                                1 + len1 + len2)
                            self.superparagraphs[found_superpara + 1].start = (
                                self.superparagraphs[found_superpara].start)
                            del self.superparagraphs[found_superpara]

                        elif (found_superpara > 0 and
                            not self.superparagraphs[found_superpara - 1].new_line):
                            # We merge with previous superpara
                            self.superparagraphs[found_superpara - 1].length += (
                                1 + len1 + len2)
                            del self.superparagraphs[found_superpara]

                        else:
                            # Do not merge with other paras
                            self.superparagraphs[found_superpara].new_line = False
                            self.superparagraphs[found_superpara].id = (
                                Paragraph.compute_para_id("?"))

                # Inserting new_line inside text superpara
                elif (not self.superparagraphs[found_superpara].new_line and
                    para.new_line):
                    # We need to split the text superpara where we insert
                    # the new_line para
                    orig_id = self.superparagraphs[found_superpara].id
                    new_ids = Paragraph.compute_para_id("split", orig_id)
                    sp_start1 = self.superparagraphs[found_superpara].start
                    sp_last1 = idx - 1
                    sp_start2 = idx + 1

                    if idx < len(self.paragraphs) and self.paragraphs[idx].new_line:
                        sp_start2 = sp_start2 + 1
                        sp_length1 = sp_last1 - sp_start1 + 1
                        sp_length2 = (self.superparagraphs[found_superpara].length -
                            2 - sp_length1)
                    else:
                        assert ignore_assertions or self.paragraphs[idx - 1].new_line
                        sp_last1 = sp_last1 - 1
                        sp_length1 = sp_last1 - sp_start1 + 1
                        sp_length2 = (self.superparagraphs[found_superpara].length -
                            2 - sp_length1)

                    new_super_para_nl = SuperParagraph(
                        sp_start1 + sp_length1, 2, True, 'x')
                    new_super_para = SuperParagraph(
                        sp_start1, sp_length1, False, new_ids[0])

                    self.superparagraphs[found_superpara].id = new_ids[1]
                    self.superparagraphs[found_superpara].length = sp_length2
                    self.superparagraphs[found_superpara].start = sp_start2

                    if sp_length2 == 0:
                        del self.superparagraphs[found_superpara]
                        new_super_para.id = orig_id # Because we are not splitting
                    self.superparagraphs.insert(found_superpara, new_super_para_nl)

                    if sp_length1 == 0:
                        # Because we are not splitting
                        self.superparagraphs[found_superpara + 1].id = orig_id
                    else:
                        self.superparagraphs.insert(found_superpara, new_super_para)

        self.paragraphs.insert(idx, para)

    def get_superparagraph_id(self, idx, return_idx=False):
        """
        Get the id of the superparagraph that contains the paragraph in
        position "idx" of self.paragraphs.
        :param idx: index of the paragraph in self.paragraphs
        :param return_idx: boolean whether to return the superpara's index
        """
        for sidx, superpara in enumerate(self.superparagraphs):
            if (idx >= superpara.start and
                idx < superpara.start + superpara.length):
                break
        else:
            assert False # Shouldn't happen

        if not return_idx:
            return superpara.id
        else:
            return superpara.id, sidx

    def create_paragraphs_from_ops(self, new_elem_ops_sorted,
        ignore_assertions=False):
        """
        Build the paragraphs for the pad based on the existing paragraphs
        and the new elementary operations
        :param new_elem_ops_sorted: list of elementary operation for the
            pad that we want to add to the paragraphs to
        """

        def para_it_belongs(elem_op_to_look_for):
            """
            Return the index of the paragraph the elem_op should belong to.
            Returns -1 if it doesn't belong to any paragraph
            :param elem_op_to_look_for: elem op we are looking at
            :return: paragraph index, number_new_lines
            """
            number_new_lines = 0
            for para_i, paragraph in enumerate(self.paragraphs):
                if ((not paragraph.new_line) and
                    paragraph.abs_position <= elem_op_to_look_for.abs_position <=
                    paragraph.abs_position + paragraph.get_length()):
                        return para_i, number_new_lines
                if paragraph.new_line:
                    number_new_lines += 1
            return -1, number_new_lines

        # Look at each elem_op and assign it to a new/existing paragraph
        for elem_op in new_elem_ops_sorted:
            # From where we will change the paragraph indices
            # should be infinity but this is enough since we can't add more
            # than 2 paragraphs
            if self.start_time > elem_op.timestamp:
                self.start_time = elem_op.timestamp
            update_indices_from = len(self.paragraphs) + 3

            # If it is a new line, create a new paragraph and insert it at
            # the right place
            if elem_op.operation_type == "add" and "\n" in elem_op.text_to_add:
                # To which paragraph this op corresponds?
                para_it_belongs_to, number_new_lines = para_it_belongs(elem_op)

                # If we found it, compute the "absolute index"
                # (i.e. the index of the paragraph in all_paragraphs)
                para_it_belongs_to_absolute = -1
                if para_it_belongs_to != -1:
                    para_it_belongs_to_absolute = self.get_absolute_index(
                        para_it_belongs_to)

                # If it is supposed to be in a paragraph which is a new line
                # or at the end of paragraph
                if para_it_belongs_to == -1:
                    # Is it an op at the beginning?
                    if elem_op.abs_position == 0:
                        if len(self.all_paragraphs):
                            new_paragraph_id = Paragraph.compute_para_id("insert_before",
                                self.all_paragraphs[0].paragraph_id)
                        else:
                            new_paragraph_id = Paragraph.compute_para_id("initial")

                        new_paragraph = Paragraph(elem_op, new_line=True,
                            paragraph_id=new_paragraph_id)

                        self.insert_paragraphs(0, new_paragraph,
                            ignore_assertions=ignore_assertions)
                        self.all_paragraphs.insert(0, new_paragraph)

                        elem_op.assign_para(2 * [[0]])
                        elem_op.assign_paragraph_id(new_paragraph_id)
                        got_superpara_id, got_sidx = self.get_superparagraph_id(
                            0, return_idx=True)
                        update_indices_from = 1

                    # Is it the last op?
                    elif (self.paragraphs[-1].abs_position +
                        self.paragraphs[-1].length <=
                        elem_op.abs_position):

                        if len(self.all_paragraphs):
                            new_paragraph_id = Paragraph.compute_para_id("insert_after",
                                self.all_paragraphs[-1].paragraph_id)
                        else:
                            new_paragraph_id = Paragraph.compute_para_id("initial")

                        new_paragraph = Paragraph(elem_op, new_line=True,
                            paragraph_id=new_paragraph_id)

                        self.insert_paragraphs(len(self.paragraphs),
                            new_paragraph,
                            ignore_assertions=ignore_assertions)

                        self.all_paragraphs.append(new_paragraph)

                        elem_op.assign_para(2*
                            [[number_new_lines - number_new_lines - 1]])
                        elem_op.assign_paragraph_id(new_paragraph_id)
                        got_superpara_id, got_sidx = self.get_superparagraph_id(
                            len(self.paragraphs) - 1, return_idx=True)

                    # Find where we should insert it
                    else:
                        para_idx = 0
                        number_new_lines = 0
                        while (self.paragraphs[para_idx].abs_position +
                            self.paragraphs[para_idx].length <
                            elem_op.abs_position):

                            para_idx += 1
                            if self.paragraphs[para_idx].new_line:
                                number_new_lines += 1
                        para_idx_absolute = self.get_absolute_index(para_idx)

                        if para_idx_absolute + 1 < len(self.all_paragraphs):
                            new_paragraph_id = Paragraph.compute_para_id("insert_between",
                                self.all_paragraphs[para_idx_absolute].paragraph_id,
                                self.all_paragraphs[para_idx_absolute + 1].paragraph_id)
                        else:
                            new_paragraph_id = Paragraph.compute_para_id("insert_after",
                                self.all_paragraphs[para_idx_absolute].paragraph_id)


                        new_paragraph = Paragraph(elem_op, new_line=True,
                            paragraph_id=new_paragraph_id)

                        # Insert it
                        self.insert_paragraphs(para_idx + 1,
                            new_paragraph,
                            ignore_assertions=ignore_assertions)
                        self.all_paragraphs.insert(para_idx_absolute + 1, new_paragraph)

                        # TODO: Make sure that `para_idx - number_new_lines + 1` exists
                        elem_op.assign_para(2 * [[para_idx - number_new_lines,
                                            para_idx - number_new_lines + 1]])
                        elem_op.assign_paragraph_id(new_paragraph_id)
                        got_superpara_id, got_sidx = self.get_superparagraph_id(
                            para_idx + 1, return_idx=True)
                        update_indices_from = para_idx + 2

                # or if it is at the start of a non-newline para:
                elif (self.paragraphs[para_it_belongs_to].abs_position ==
                    elem_op.abs_position):
                    if para_it_belongs_to_absolute - 1 >= 0:
                        new_paragraph_id = Paragraph.compute_para_id("insert_between",
                            self.all_paragraphs[para_it_belongs_to_absolute - 1].paragraph_id,
                            self.all_paragraphs[para_it_belongs_to_absolute].paragraph_id)
                    else:
                        new_paragraph_id = Paragraph.compute_para_id("insert_before",
                            self.all_paragraphs[para_it_belongs_to_absolute].paragraph_id)

                    new_paragraph = Paragraph(elem_op, new_line=True,
                        paragraph_id=new_paragraph_id)

                    self.insert_paragraphs(para_it_belongs_to,
                        new_paragraph,
                        ignore_assertions=ignore_assertions)
                    self.all_paragraphs.insert(para_it_belongs_to_absolute,
                                               new_paragraph)

                    elem_op.assign_para(2 *
                        [[para_it_belongs_to - number_new_lines - 1,
                            para_it_belongs_to - number_new_lines]])
                    elem_op.assign_paragraph_id(new_paragraph_id)
                    got_superpara_id, got_sidx = self.get_superparagraph_id(
                        para_it_belongs_to, return_idx=True)
                    update_indices_from = para_it_belongs_to + 1

                # or if it is at the end of a non-newline para:
                elif (self.paragraphs[para_it_belongs_to].abs_position +
                    self.paragraphs[para_it_belongs_to].length ==
                    elem_op.abs_position):

                    if para_it_belongs_to_absolute + 1 < len(self.all_paragraphs):
                        new_paragraph_id = Paragraph.compute_para_id("insert_between",
                            self.all_paragraphs[para_it_belongs_to_absolute].paragraph_id,
                            self.all_paragraphs[para_it_belongs_to_absolute + 1].paragraph_id)
                    else:
                        new_paragraph_id = Paragraph.compute_para_id("insert_after",
                            self.all_paragraphs[para_it_belongs_to_absolute].paragraph_id)

                    new_paragraph = Paragraph(elem_op, new_line=True,
                        paragraph_id=new_paragraph_id)

                    self.insert_paragraphs(para_it_belongs_to + 1,
                        new_paragraph,
                        ignore_assertions=ignore_assertions)
                    self.all_paragraphs.insert(para_it_belongs_to_absolute + 1,
                                               new_paragraph)

                    elem_op.assign_para(2 *
                        [[para_it_belongs_to - number_new_lines,
                            para_it_belongs_to - number_new_lines + 1]])
                    elem_op.assign_paragraph_id(new_paragraph_id)
                    got_superpara_id, got_sidx = self.get_superparagraph_id(
                        para_it_belongs_to + 1, return_idx=True)
                    update_indices_from = para_it_belongs_to + 2

                # We split the paragraph in two, where there is the newline
                else:
                    # The two paragraphs from the split
                    para1, para2, para_new_line_id = Paragraph.split(
                        self.paragraphs[para_it_belongs_to],
                        elem_op.abs_position,
                        return_new_line_paragraph_id=True)

                    # We delete the old paragraph
                    self.delete_paragraphs(para_it_belongs_to,
                        abs_idx=para_it_belongs_to_absolute,
                        action="split",
                        ignore_assertions=ignore_assertions)
                    # Insert the second paragraph
                    self.paragraphs.insert(para_it_belongs_to, para2)
                    self.all_paragraphs.insert(para_it_belongs_to_absolute, para2)
                    # Insert the new line just before
                    para_new_line = Paragraph(elem_op, new_line=True,
                                              paragraph_id=para_new_line_id)
                    self.paragraphs.insert(para_it_belongs_to, para_new_line)
                    self.all_paragraphs.insert(para_it_belongs_to_absolute, para_new_line)
                    # Insert the first paragraph
                    self.paragraphs.insert(para_it_belongs_to, para1)
                    self.all_paragraphs.insert(para_it_belongs_to_absolute, para1)

                    elem_op.assign_para([
                        [para_it_belongs_to - number_new_lines],
                        [para_it_belongs_to - number_new_lines,
                            para_it_belongs_to - number_new_lines + 1]])
                    elem_op.assign_paragraph_id(para1.paragraph_id)
                    got_superpara_id, got_sidx = self.get_superparagraph_id(
                        para_it_belongs_to, return_idx=True)

                    # From where we will update the indices
                    update_indices_from = para_it_belongs_to + 3

                elem_op.assign_superparagraph_id(got_superpara_id)
                n_authors = self.superparagraphs[got_sidx].add_author(elem_op.author)
                elem_op.assign_number_coauthors(n_authors)

                # We need to notify the paragraphs that are after my edit,
                # that their position might have changed
                for para in self.paragraphs[update_indices_from:]:
                    para.update_indices(elem_op)

            # If it is a deletion
            elif elem_op.operation_type == "del":
                # For all paragraphs add the elem_op if it affects them partly
                # or delete them if it affects them totally.
                # It is also possible that we have to merge the paragraph on
                # the extremities if no new_lines are in between
                # Index of the left paragraph to merge
                merge1 = None
                # Index of the right paragraph to merge
                merge2 = None
                # Paragraphs to remove
                to_remove = []

                number_new_lines = 0
                para_to_assign_before = []
                para_to_assign_after = []

                # We will check for each paragraph if they are concerned
                for para_idx, para in enumerate(self.paragraphs):
                    if para.new_line:
                        number_new_lines += 1

                    # We are deleting only this paragraph
                    if (elem_op.abs_position == para.abs_position and
                        elem_op.abs_position + elem_op.length_to_delete ==
                        para.abs_position + para.get_length()):

                        # Add to the list to remove
                        to_remove.append(para_idx)
                        # We will update the indices from here
                        update_indices_from = para_idx + 1

                        para_to_assign_before = [para_idx - number_new_lines,
                                              para_idx - number_new_lines + 1]
                        para_to_assign_after = [para_idx - number_new_lines,
                                             para_idx - number_new_lines + 1]

                        # If the paragraph just before is not a new line, we
                        # might merge it with the one after
                        if (para_idx > 0 and
                            not self.paragraphs[para_idx - 1].new_line and
                            para.new_line and
                            merge1 is None):
                            # Paragraph just before which will merge if there
                            # is a merge
                            merge1 = para_idx - 1

                        # If the paragraph just after is not a new line
                        # if we are merging (merge1 is not None -> the paragraph
                        # before was not a new line), then we will merge
                        if (para_idx < len(self.paragraphs) - 1 and
                            (not self.paragraphs[para_idx + 1].new_line) and
                            merge1 is not None):
                            # Paragraph just after which will merge with the
                            # paragraph at merge1
                            merge2 = para_idx + 1
                            para_to_assign_after = [para_idx - number_new_lines]

                        if para_idx == len(self.paragraphs) - 1:
                            para_to_assign_before = [para_idx - number_new_lines]
                            para_to_assign_after = [para_idx - number_new_lines]
                        elif para.new_line:
                            para_to_assign_before = [para_idx - number_new_lines]
                            para_to_assign_after = [
                                para_idx - number_new_lines - 1,
                                para_idx - number_new_lines]

                        elem_op.assign_para([para_to_assign_before, para_to_assign_after])

                        # We delete only this paragraph
                        break

                    # paragraph is fully contained in deletion and it touches
                    # other paragraphs
                    elif (elem_op.abs_position <= para.abs_position and
                        elem_op.abs_position + elem_op.length_to_delete >=
                        para.abs_position + para.get_length()):
                        # We remove the whole paragraph
                        to_remove.append(para_idx)

                        # shouldn't be necessary since it will be taken care
                        # by the last paragraph we delete.
                        update_indices_from = para_idx + 1

                        # If we are a new line and before us was text and we
                        # are currently not merging (this means we are the
                        # first op considering merging), we might be merging
                        # with the paragraph just before.
                        # This is usually when the current paragraph is the
                        # start of the deletion
                        if (0 < para_idx and
                            para.new_line and
                            (not self.paragraphs[para_idx - 1].new_line) and
                            merge1 is None):
                            # The paragraph just before that might merge
                            merge1 = para_idx - 1

                        # If we are considering merging and the paragraph just
                        # after is not a new line, then it might
                        # be the second part of the merge
                        if (para_idx < len(self.paragraphs) - 1 and
                            merge1 is not None and
                            (not self.paragraphs[para_idx + 1].new_line)):
                            # The paragraph just after that we might merge with
                            merge2 = para_idx + 1
                        else:
                            # If the paragraph just after is not a possible
                            # merger then reset the right en of the merge
                            merge2 = None

                        if not len(para_to_assign_before):
                            # If it's the first para that we are assigning to op.
                            # Not completely accurate, because we may be deleting
                            # one new_line followed by a text paragraph, and it
                            # will add the para that is before new_line
                            para_to_assign_before.append(para_idx - number_new_lines)
                            para_to_assign_after.append(para_idx - number_new_lines)

                    # Start of deletion is within our para whether the end is
                    # within para or not
                    elif (para.abs_position <=
                        elem_op.abs_position <
                        para.abs_position + para.get_length()):
                        # Add the operation to the para
                        para.add_elem_op(elem_op)
                        # update the indices from here
                        update_indices_from = para_idx + 1
                        # If there is a merge, it will be from here
                        merge1 = para_idx

                        if not len(para_to_assign_before):
                            # If it's the first para that we are assigning to op.
                            # Not completely accurate, because we may be deleting
                            # one new_line followed by a text paragraph, and it
                            # will add the para that is before new_line
                            para_to_assign_before.append(para_idx - number_new_lines)
                            para_to_assign_after.append(para_idx - number_new_lines)

                    # End of deletion is within our para but start isn't
                    # (or it would have gone in the elif before
                    elif (para.abs_position <
                        elem_op.abs_position + elem_op.length_to_delete <=
                        para.abs_position + para.get_length()):
                        # If there is no merge, apply the op to the paragraph.
                        # Otherwise, it is included in the first merge
                        if merge1 is None:
                            # Add the elem_op
                            para.add_elem_op(elem_op)
                            # Update the indices from here
                            update_indices_from = para_idx + 1
                        # We will merge with this paragraph
                        # (if there is a merge, aka. merge1 is not None)
                        merge2 = para_idx

                        para_to_assign_before.append(para_idx - number_new_lines)

                    # paragraph is not concerned
                    else:
                        pass

                elem_op.assign_para([para_to_assign_before, para_to_assign_after])

                # Check that if the start of the deletion is withing a para
                # and the end of the deletion is within another para.
                # If so we merge the two paragraphs.
                if (merge1 is not None and
                    merge2 is not None and
                    not (merge1 in to_remove or merge2 in to_remove)):
                    # Merged paragraph
                    merged_paragraph = Paragraph.merge(self.paragraphs[merge1],
                       self.paragraphs[merge2], elem_op)
                    # Get superpara ids before removing anything
                    sp_id1, sp_idx1 = self.get_superparagraph_id(merge1,
                        return_idx=True)
                    sp_id2, sp_idx2 = self.get_superparagraph_id(merge2,
                        return_idx=True)

                    if sp_id1 != sp_id2:
                        sp_id = Paragraph.compute_para_id("merge", sp_id1, sp_id2)
                        self.superparagraphs[sp_idx1].id = sp_id
                        sp_id1 = sp_id

                    # We remove the second paragraph
                    merge2_absolute = self.get_absolute_index(merge2)
                    self.delete_paragraphs(merge2, merge2_absolute,
                        ignore_assertions=ignore_assertions)
                    # We put the merged paragraph where the first paragraph was
                    self.paragraphs[merge1] = merged_paragraph
                    merge1_absolute = self.get_absolute_index(merge1)
                    self.all_paragraphs[merge1_absolute].is_deleted = True
                    self.all_paragraphs.insert(merge1_absolute, merged_paragraph)
                    # We will update the indices from the paragraph from here
                    update_indices_from = merge2

                    elem_op.assign_paragraph_id(merged_paragraph.paragraph_id)
                    elem_op.assign_superparagraph_id(sp_id1)
                    n_authors = self.superparagraphs[sp_idx1].add_author(elem_op.author)
                    elem_op.assign_number_coauthors(n_authors)

                # We need to notify the paragraphs that are after my edit,
                # that their position might have changed
                for para in self.paragraphs[update_indices_from:]:
                    para.update_indices(elem_op)

                # Remove the paragraphs we are supposed to remove
                for idx in to_remove[::-1]:
                    elem_op.assign_paragraph_id(self.paragraphs[idx].paragraph_id)
                    got_superpara_id, got_sidx = self.get_superparagraph_id(
                        idx, return_idx=True)
                    elem_op.assign_superparagraph_id(got_superpara_id)
                    n_authors = self.superparagraphs[got_sidx].add_author(elem_op.author)
                    elem_op.assign_number_coauthors(n_authors)

                    idx_absolute = self.get_absolute_index(idx)
                    self.delete_paragraphs(idx, idx_absolute,
                        ignore_assertions=ignore_assertions)

            # Keep our paragraphs as they are, just add the elem_op
            else:
                # Find the paragraph it should belong to
                para_it_belongs_to, number_new_lines = para_it_belongs(elem_op)

                # If we found it, compute the "absolute index"
                # (i.e. the index of the paragraph in all_paragraphs)
                para_it_belongs_to_absolute = -1
                if para_it_belongs_to != -1:
                    para_it_belongs_to_absolute = self.get_absolute_index(
                        para_it_belongs_to)

                # If we should create a new para for this elem_op
                if para_it_belongs_to == -1:
                    # If we are at the start of the document
                    if elem_op.abs_position == 0:
                        if len(self.all_paragraphs):
                            new_paragraph_id = Paragraph.compute_para_id("insert_before",
                                self.all_paragraphs[0].paragraph_id)
                        else:
                            new_paragraph_id = Paragraph.compute_para_id("initial")

                        new_paragraph = Paragraph(elem_op,
                            paragraph_id=new_paragraph_id)

                        self.insert_paragraphs(0, new_paragraph,
                            ignore_assertions=ignore_assertions)
                        self.all_paragraphs.insert(0, new_paragraph)

                        elem_op.assign_para(2 * [[0]])
                        elem_op.assign_paragraph_id(new_paragraph_id)
                        got_superpara_id, got_sidx = self.get_superparagraph_id(
                            0, return_idx=True)

                        update_indices_from = 1

                    # If we are at the end of the paragraph
                    elif (self.paragraphs[-1].abs_position +
                        self.paragraphs[-1].length <=
                        elem_op.abs_position):

                        if len(self.all_paragraphs):
                            new_paragraph_id = Paragraph.compute_para_id("insert_after",
                                self.all_paragraphs[-1].paragraph_id)
                        else:
                            new_paragraph_id = Paragraph.compute_para_id("initial")

                        new_paragraph = Paragraph(elem_op,
                            paragraph_id=new_paragraph_id)

                        self.insert_paragraphs(len(self.paragraphs),
                            new_paragraph,
                            ignore_assertions=ignore_assertions)
                        self.all_paragraphs.append(new_paragraph)

                        elem_op.assign_para([
                            [len(self.paragraphs) - number_new_lines - 1],
                            [len(self.paragraphs) - number_new_lines]])
                        elem_op.assign_paragraph_id(new_paragraph_id)
                        got_superpara_id, got_sidx = self.get_superparagraph_id(
                            len(self.paragraphs) - 1, return_idx=True)

                    # Insert the new paragraph in the good place
                    else:
                        # Look where we should insert it
                        para_idx = 0
                        number_new_lines = 0
                        while (self.paragraphs[para_idx].abs_position +
                            self.paragraphs[para_idx].length <
                            elem_op.abs_position):

                            para_idx += 1
                            if self.paragraphs[para_idx].new_line:
                                number_new_lines += 1
                        para_idx_absolute = self.get_absolute_index(para_idx)

                        if para_idx_absolute + 1 < len(self.all_paragraphs):
                            new_paragraph_id = Paragraph.compute_para_id("insert_between",
                                self.all_paragraphs[para_idx_absolute].paragraph_id,
                                self.all_paragraphs[para_idx_absolute + 1].paragraph_id)
                        else:
                            new_paragraph_id = Paragraph.compute_para_id("insert_after",
                                self.all_paragraphs[para_idx_absolute].paragraph_id)

                        new_paragraph = Paragraph(elem_op,
                            paragraph_id=new_paragraph_id)

                        # Insert it
                        self.insert_paragraphs(para_idx + 1,
                            new_paragraph,
                            ignore_assertions=ignore_assertions)
                        self.all_paragraphs.insert(para_idx_absolute + 1, new_paragraph)

                        elem_op.assign_para([
                            [para_idx - number_new_lines, para_idx - number_new_lines + 1],
                            [para_idx - number_new_lines + 1]])
                        elem_op.assign_paragraph_id(new_paragraph_id)
                        got_superpara_id, got_sidx = self.get_superparagraph_id(
                            para_idx + 1, return_idx=True)

                        # Update indices of the next paragraphs from here
                        update_indices_from = para_idx + 2

                # Just add to the paragraph
                else:
                    elem_op.assign_para([
                        [para_it_belongs_to - number_new_lines],
                        [para_it_belongs_to - number_new_lines]])
                    elem_op.assign_paragraph_id(
                        self.paragraphs[para_it_belongs_to].paragraph_id)

                    got_superpara_id, got_sidx = self.get_superparagraph_id(
                        para_it_belongs_to, return_idx=True)

                    # Add it
                    self.paragraphs[para_it_belongs_to].add_elem_op(elem_op)
                    # Update indices of the next paragraphs from here
                    update_indices_from = para_it_belongs_to + 1

                elem_op.assign_superparagraph_id(got_superpara_id)
                n_authors = self.superparagraphs[got_sidx].add_author(elem_op.author)
                elem_op.assign_number_coauthors(n_authors)

                # We need to notify the paragraphs that are after my edit,
                # that their position might have changed
                for para in self.paragraphs[update_indices_from:]:
                    para.update_indices(elem_op)

            # Assertions
            # TODO remove for production
            if not ignore_assertions:
                # Checking that the paragraph is in order
                assert self.paragraphs == sorted(self.paragraphs)
                # checking that length is not 0
                for i in range(0, len(self.paragraphs)):
                    if self.paragraphs[i].length == 0:
                        raise AssertionError
                # Checking that the paragraphs touch each others
                for i in range(1, len(self.paragraphs)):
                    if self.paragraphs[i - 1].abs_position \
                            + self.paragraphs[i - 1].length \
                            != self.paragraphs[i].abs_position:
                        # print(elem_op)
                        raise AssertionError
                # Checking that a text paragraph is encapsulated between two
                # new_line paragraphs
                for i in range(1, len(self.paragraphs) - 1):
                    if not (self.paragraphs[i].new_line or
                            self.paragraphs[i + 1].new_line):
                        # print(elem_op)
                        raise AssertionError

        # Find the list of authors in the pad
        for op in self.operations:
            if op.author not in self.authors:
                self.authors.append(op.author)

    def classify_operations(self, length_edit, length_delete):
        """
        Classify all the operations types from the pad.
        The different types are Write, Edit, Delete, Copy/Paste or Jump.
        :param length_edit: Threshold in length to differentiate a Write type
            from an Edit or an Edit from a Deletion.
        :param length_delete:  Threshold in length to consider the op as a deletion
        :return: None
        """
        for op in self.operations:
            # Classify the type according to the length of the operation
            len_op = op.get_length_of_op()
            if len_op >= length_edit:
                if len(op.elem_ops) == 1:
                    op.type = 'paste'
                else:
                    op.type = 'write'
            elif len_op <= -length_delete:
                op.type = 'delete'
            elif (len(op.elem_ops) == 1 and
                op.elem_ops[0].operation_type == "add" and
                op.elem_ops[0].text_to_add == '\n'):
                op.type = 'jump'
            else:
                op.type = 'edit'

    def build_operation_context(self,
        delay_sync,
        time_to_reset_day,
        time_to_reset_break):
        """
        Build the context of each operation progressively added to the pad.
        The context is a dictionary containing whether a pad is synchronous
        wih an other author in the pad or in the paragraph and it contains
        list of authors accordingly.
        :param delay_sync: delay of synchronization between two authors
        :param time_to_reset_day: Number of milliseconds between two ops to
            indicate the first op of the day, by default 8h
        :param time_to_reset_break: Number of milliseconds to indicate the
            first op after a break, by default 10min
        :return: None
        """
        # Iterate over all Operation of each Paragraph which is the same as
        # to iterate all iterations of the pad
        pad_operations = self.operations
        len_pad = sum([abs(op.get_length_of_op()) for op in pad_operations])

        for op in pad_operations:
            # Initialize the context
            len_op = abs(op.get_length_of_op())
            op.context['proportion_pad'] = len_op / len_pad
            # An operation is originally 100% of a new paragraph
            op.context['proportion_paragraph'] = 1
            op.context['synchronous_in_pad'] = False
            op.context['synchronous_in_pad_with'] = []
            op.context['synchronous_in_paragraph'] = False
            op.context['synchronous_in_paragraph_with'] = []
            op.context['first_op_day'] = False
            op.context['first_op_break'] = False
            start_time = op.timestamp_start
            end_time = op.timestamp_end

            # Check in the pad if the other operations are written by someone
            # else at the same time (+ some delay)
            op_index = 0
            for other_op in pad_operations:
                other_start_time = other_op.timestamp_start
                # Control if this is the current operation to do some processing on it
                if other_op == op:
                    # Check if the op is a first one
                    if op_index == 0 or other_start_time >= pad_operations[
                                op_index - 1].timestamp_end + time_to_reset_day:
                        op.context['first_op_day'] = True
                    elif op_index != 0 and other_start_time >= pad_operations[
                                op_index - 1].timestamp_end + time_to_reset_break:
                        op.context['first_op_break'] = True
                op_index += 1

                if (other_op.author != op.author and
                    other_op.author != 'Etherpad_admin' and
                    op.author != 'Etherpad_admin' and
                    end_time + delay_sync >= other_start_time >= start_time - delay_sync):
                    op.context['synchronous_in_pad'] = True
                    if other_op.author not in op.context['synchronous_in_pad_with']:
                        op.context['synchronous_in_pad_with'].append(other_op.author)

        for para in self.paragraphs:
            abs_length_para = 0
            para_ops = para.operations

            for op in para_ops:
                # Initialize the variables
                len_op = abs(op.get_length_of_op())
                start_time = op.timestamp_start
                end_time = op.timestamp_end

                # Compute the overall length of the paragraph
                abs_length_para += abs(op.get_length_of_op())

                for other_op in para_ops:
                    other_start_time = other_op.timestamp_start
                    if (other_op.author != op.author and
                        other_op.author != 'Etherpad_admin' and
                        op.author != 'Etherpad_admin' and
                        end_time + delay_sync >= other_start_time >=
                        start_time - delay_sync):
                        op.context['synchronous_in_paragraph'] = True
                        if (other_op.author not in
                            op.context['synchronous_in_paragraph_with']):
                            op.context['synchronous_in_paragraph_with'].append(
                                other_op.author
                            )

                op.context['proportion_paragraph'] = len_op

            # Once we computed the absolute length of the paragraph, we
            # compute the proportion (it is positive)
            for op in para_ops:
                if abs_length_para != 0:
                    op.context['proportion_paragraph'] /= abs_length_para
                else:
                    op.context['proportion_paragraph'] = 0

    ###############################
    # Visualizations
    ###############################
    def get_text(self, until_timestamp=None):
        """
        Return a string with the whole text
        :param until_timestamp:
        :return: the text written so far on the pad
        :rtype: str
        """
        elem_ops_ordered = self.get_elem_ops(sorted_=True)
        text = ""
        for elem_id, elem_op in enumerate(elem_ops_ordered):
            if until_timestamp is not None and elem_op.timestamp > until_timestamp:
                return text
            if elem_op.operation_type == 'add':
                # We add to the end of the ext
                if ('*' in elem_op.text_to_add or '*' in text or
                    len(elem_ops_ordered) - 1 == elem_id):
                    pass
                if len(text) == elem_op.abs_position:
                    text += elem_op.text_to_add
                else:
                    text = (text[:elem_op.abs_position] +
                            elem_op.text_to_add +
                            text[elem_op.abs_position:])
            elif elem_op.operation_type == 'del':
                text = (text[:elem_op.abs_position] +
                        text[elem_op.abs_position +
                        elem_op.length_to_delete:])
            else:
                raise AttributeError("Undefined elementary operation")
        return text

    def display_text_colored_by_ops(self):
        """
        Print the colored text according to the operations.
        """
        letters = []
        letters_color = []
        idx_color = 0
        elem_ops_ordered = self.get_elem_ops(sorted_=True)
        op_to_color = {}
        colors = get_colors()
        for idx, elem_op in enumerate(elem_ops_ordered):
            idx_elem = elem_op.abs_position
            if elem_op.operation_type == 'add':
                # We find the right color to print
                op = elem_op.belong_to_operation
                if op in op_to_color:
                    color = op_to_color[op]
                else:
                    op_to_color[op] = colors[idx_color % len(colors)]
                    idx_color += 1
                    color = op_to_color[op]

                # We add to the end of the ext
                if idx_elem == len(letters):
                    letters += list(elem_op.text_to_add)
                    letters_color += [color] * elem_op.get_length_of_op()
                else:
                    letters = (letters[:idx_elem] +
                               list(elem_op.text_to_add) +
                               letters[idx_elem:])
                    letters_color = (letters_color[:idx_elem] +
                                     [color] * elem_op.get_length_of_op() +
                                     letters_color[idx_elem:])
            elif elem_op.operation_type == 'del':
                letters = (letters[:idx_elem] +
                           letters[idx_elem +
                           elem_op.length_to_delete:])
                letters_color = (letters_color[:idx_elem] +
                                 letters_color[idx_elem +
                                 elem_op.length_to_delete:])
            else:
                raise AttributeError("Undefined elementary operation")
        # Print letter after letter with the right color
        string_colored = ''.join([color + letter for letter, color
                                  in zip(letters, letters_color)])

        # Change color back to original at the end and return
        return string_colored + get_colors()[0]

    def get_letters_and_colors_from_authors(self):
        """
        Create one list for all the letters representing the text and one list
        with the color of each letter according to its author.
        :return: list of letters representing the text and list of colors
            representing authors
        :rtype: list[str], list[str]
        """
        letters = []
        letters_color = []
        elem_ops_ordered = self.get_elem_ops(sorted_=True)
        authors = self.authors
        colors = get_colors()
        for elem_op in elem_ops_ordered:
            idx_elem = elem_op.abs_position

            color = colors[0]
            if elem_op.author in authors:
                color = colors[authors.index(elem_op.author) % len(colors)]
            if elem_op.operation_type == 'add':
                # We add to the end of the ext
                if idx_elem == len(letters):
                    letters += list(elem_op.text_to_add)
                    letters_color += [color] * elem_op.get_length_of_op()
                else:
                    letters = (letters[:idx_elem] +
                               list(elem_op.text_to_add) +
                               letters[idx_elem:])
                    letters_color = (letters_color[:idx_elem] +
                                     [color] * elem_op.get_length_of_op() +
                                     letters_color[idx_elem:])
            elif elem_op.operation_type == 'del':
                letters = (letters[:idx_elem] +
                           letters[idx_elem +
                           elem_op.length_to_delete:])
                letters_color = (letters_color[:idx_elem] +
                                 letters_color[idx_elem +
                                 elem_op.length_to_delete:])
            else:
                raise AttributeError("Undefined elementary operation")
        return letters, letters_color

    def display_text_colored_by_authors(self):
        """
        Display the text the same way as get_text but with
        different colors according to authors.
        :return: None
        """
        letters, colors = self.get_letters_and_colors_from_authors()

        # Print letter after letter with the right color
        colored_text = ''.join([color + letter for letter, color
                                in zip(letters, colors)])

        # Change color back to original
        return colored_text + get_colors()[0]

    def display_csv(self,
        separator_char='\t', string_delimiter='"',
        pad_id=None):
        """
        Print all operations' descriptors in csv format, with one line per
        operation.
        :param separator_char: separation between columns
        :param string_delimiter: string delimiter
        :param pad_id: pad id to be used. If none, the available one is used
        """
        for op in self.operations:
            if pad_id is None:
                print(self.pad_name + separator_char +
                      op.get_line(separator_char=separator_char,
                                  string_delimiter=string_delimiter))
            else:
                print(format(pad_id) + separator_char +
                      op.get_line(separator_char=separator_char,
                                  string_delimiter=string_delimiter))

    def display_operations(self):
        """
        Print the descriptions of all the operations done on the pad
        """
        for op in self.operations:
            print(op)
            print()

    def display_paragraphs(self, verbose=0):
        """
        Print all the paragraphs contained in the pad
        :param verbose: print detailed info if verbose is 1
        :return:
        """
        for para in self.paragraphs:
            print(para.__str__(verbose))
            print("\n")

    def get_paragraphs_text(self, splitter=None, remove_default=True):
        """
        Returns an array containing one string for each of the paragraphs in
        the pad. If splitter is specified, it is used for splitting the
        paragraphs (e.g. '\n', '\n\n', '\n\n\n'...).
        If no splitter is specified, the function tries to split by double
        new lines. If with this splitting the average length of the paragraphs
        obtained is longer than config.max_length_paragraph, paragraphs are
        split by single new lines instead.
        :param splitter: characters that separate paragraphs (e.g. '\n')
        :return: dictionary with text split by paragraphs
        """
        all_text = self.get_text()
        if remove_default:
            for default_line in config.default_lines:
                if default_line not in all_text:
                    break
                else:
                    all_text = all_text.replace(default_line, "", 1)

        if splitter is None:
            superparagraphs_text = [
                superpara.strip() for superpara in
                all_text.split("\n\n") if
                superpara.strip() != ""]
            mean_size = np.mean([len(sp) for sp in superparagraphs_text])

            if mean_size > config.max_length_paragraph:
                superparagraphs_text = [
                    superpara.strip() for superpara in
                    all_text.split("\n") if
                    superpara.strip() != ""]

        else:
            superparagraphs_text = [
                superpara.strip() for superpara in
                all_text.split(splitter) if
                superpara.strip() != ""]

        return superparagraphs_text

    ###############################
    # Metrics
    ###############################
    def compute_metrics(self, start_time=0):
        """
        Computes all pad metrics and returns them in a dictionary.
        :param start_time: start time from which to consider operations
        :return: dictionary with metrics
        """
        # Obtain dictionaries with several values for better efficiency
        type_overall_score_dict = self.type_overall_score(None)
        window_type_overall_score_dict = self.type_overall_score(
            None, start_time=start_time)

        user_type_score_dict = self.user_type_score(None)
        window_user_type_score_dict = self.user_type_score(
            None, start_time=start_time)

        count_chars_dict = self.count_chars(None, start_time=start_time)

        [length_all, length_all_write, length_all_paste] = (
            self.get_all_text_added_length())

        # Generate dictionary with computed metrics
        metrics_dict = {
            # Overall metrics:
            "user_participation_paragraph_score": (
                self.user_participation_paragraph_score()),
            "prop_score": self.prop_score(),
            "sync_score": self.sync_score()[0],
            "alternating_score": self.alternating_score(),
            "break_score_day": self.break_score('day'),
            "break_score_short": self.break_score('short'),
            "type_overall_score_write": type_overall_score_dict['write'],
            "type_overall_score_paste": type_overall_score_dict['paste'],
            "type_overall_score_delete": type_overall_score_dict['delete'],
            "type_overall_score_edit": type_overall_score_dict['edit'],
            "user_type_score_write": user_type_score_dict['write'],
            "user_type_score_paste": user_type_score_dict['paste'],
            "user_type_score_delete": user_type_score_dict['delete'],
            "user_type_score_edit": user_type_score_dict['edit'],
            # Time-window metrics:
            "window_sync_score": self.sync_score(start_time=start_time)[0],
            "window_break_score_day": self.break_score('day', start_time=start_time),
            "window_break_score_short": self.break_score(
                'short', start_time=start_time),
            "window_type_overall_score_write": (
                window_type_overall_score_dict['write']),
            "window_type_overall_score_paste": (
                window_type_overall_score_dict['paste']),
            "window_type_overall_score_delete": (
                window_type_overall_score_dict['delete']),
            "window_type_overall_score_edit": (
                window_type_overall_score_dict['edit']),
            "window_user_type_score_write": (
                window_user_type_score_dict['write']),
            "window_user_type_score_paste": (
                window_user_type_score_dict['paste']),
            "window_user_type_score_delete": (
                window_user_type_score_dict['delete']),
            "window_user_type_score_edit": (
                window_user_type_score_dict['edit']),
            # Other measures
            "length": self.get_text_length(),
            "length_all": length_all,
            "length_all_write": length_all_write,
            "length_all_paste": length_all_paste,
            "added_chars": count_chars_dict['add'],
            "deleted_chars": count_chars_dict['del'],
            "paragraph_average_length": self.paragraph_average_length(),
            "superparagraph_average_length": (
                self.superparagraph_average_length()),
            "average_paragraphs_per_superparagraph": (
                self.average_paragraphs_per_superparagraph()),
        }
        return metrics_dict

    def get_metrics_text(self, metrics_dict=None, start_time=0):
        """
        Returns the metrics dictionary as a text string.
        :param metrics_dict: metrics_dict can be passed for efficiency,
            otherwise it is computed.
        :param start_time: start time from which to consider operations
        :return: string with metrics
        """
        if metrics_dict is None:
            metrics_dict = self.compute_metrics(start_time)

        metrics_text = (
            "OVERALL METRICS:\n"
            "User proportion per paragraph score: {}\n".format(
                metrics_dict["user_participation_paragraph_score"]) +
            "Proportion score: {}\n".format(
                metrics_dict["prop_score"]) +
            "Synchronous score: {}\n".format(
                metrics_dict["sync_score"]) +
            "Alternating score: {}\n".format(
                metrics_dict["alternating_score"]) +
            "Break score day: {}\n".format(
                metrics_dict["break_score_day"]) +
            "Break score short: {}\n".format(
                metrics_dict["break_score_short"]) +
            "Overall write type score: {}\n".format(
                metrics_dict["type_overall_score_write"]) +
            "Overall paste type score: {}\n".format(
                metrics_dict["type_overall_score_paste"]) +
            "Overall delete type score: {}\n".format(
                metrics_dict["type_overall_score_delete"]) +
            "Overall edit type score: {}\n".format(
                metrics_dict["type_overall_score_edit"]) +
            "User write score: {}\n".format(
                metrics_dict["user_type_score_write"]) +
            "User paste score: {}\n".format(
                metrics_dict["user_type_score_paste"]) +
            "User delete score: {}\n".format(
                metrics_dict["user_type_score_delete"]) +
            "User edit score: {}\n".format(
                metrics_dict["user_type_score_edit"]) +

            "\n\n TIME-WINDOW METRICS:\n" +
            "Added characters: {}\n".format(
                "added_chars") +
            "Deleted characters: {}\n".format(
                "deleted_chars") +
            "Paragraph average length: {}\n".format(
                "paragraph_average_length") +
            "Superparagraph average length: {}\n".format(
                "superparagraph_average_length") +
            "Average paragraphs per superparagraph: {}\n".format(
                "average_paragraphs_per_superparagraph") +
            "Window overall write score: {}\n".format(
                "window_type_overall_score_write") +
            "Window overall paste score: {}\n".format(
                "window_type_overall_score_paste") +
            "Window overall delete score: {}\n".format(
                "window_type_overall_score_delete") +
            "Window overall edit score: {}\n".format(
                "window_type_overall_score_edit") +
            "Window user write score: {}\n".format(
                "window_user_type_score_write") +
            "Window user paste score: {}\n".format(
                "window_user_type_score_paste") +
            "Window user delete score: {}\n".format(
                "window_user_type_score_delete") +
            "Window user edit score: {}\n".format(
                "window_user_type_score_edit")
        )

        return metrics_text

    def author_proportions(self, considerate_admin=True):
        """
        Compute the proportion of each author for the entire pad.
        :param considerate_admin: Boolean to determine if we include the
            admin or not in our computations
        :return: the list of authors to consider and the resulting
            proportions in an array
        :rtype: list[str], np.array[float]
        """

        # Fetch all the authors who participated in the pad
        authors = self.authors[:]

        # Delete the admin if needed
        if not considerate_admin and 'Etherpad_admin' in authors:
            authors = list(np.delete(authors, authors.index('Etherpad_admin')))
        # Initialize the number of letters written by each authors
        author_lengths = np.zeros(len(authors))
        # increment the participation accordingly
        for op in self.operations:
            op_author = op.author
            # Skip the incrementation if needed
            if considerate_admin or op_author != 'Etherpad_admin':
                author_lengths[authors.index(op_author)] += abs(op.get_length_of_op())

        # Compute the overall participation
        overall_length = sum(author_lengths)
        if overall_length:
            proportions = author_lengths / overall_length
        else:
            proportions = np.zeros(len(authors))
        return authors, proportions

    @staticmethod
    def compute_entropy_prop(proportions, len_authors):
        """
        Compute the proportion score using the entropy and proportions.
        :param proportions: list of proportions summing up to 1
        :param len_authors: number of authors collaborating
        :return: the entropy score between 0 and 1.
            0 is returned if there are less than two authors
        """
        entropy_score = 0
        # Check that we have at least 2 authors different from the admin
        if len_authors >= 2:
            # Change zero values to small values to note divide by zero
            proportions = np.array([0.000001 if prop == 0 else prop for
                                    prop in proportions])
            # Compute the entropy with the proportions
            entropy_score = (sum(np.log(1 / proportions) * proportions) /
                             np.log(len_authors))
        return entropy_score

    def prop_score(self, considerate_admin=False):
        """
        Compute the proportion score using the entropy.
        :return: proportion score between 0 and 1
        :rtype: float
        """
        authors, proportions = self.author_proportions(
            considerate_admin=considerate_admin)
        return self.compute_entropy_prop(proportions, len(authors))

    def sync_score(self, start_time=0):
        """
        Compute the synchronous and asynchronous scores.
        :return: synchronous and asynchronous scores, floats between 0 and 1.
        :rtype: (float,float)
        """
        prop_sync = 0
        prop_async = 0
        len_pad_no_admin = sum(
            [abs(op.get_length_of_op()) if op.author != 'Etherpad_admin' else 0
             for op in self.operations if op.timestamp_start >= start_time]
        )
        for op in self.operations:
            if op.timestamp_start >= start_time and len_pad_no_admin:
                    if op.context['synchronous_in_pad']:
                        prop_sync += abs(op.get_length_of_op()) / len_pad_no_admin
                    elif op.author != 'Etherpad_admin':
                        prop_async += abs(op.get_length_of_op()) / len_pad_no_admin
        return prop_sync, prop_async

    def prop_paragraphs(self):
        """
        Compute the proportion of each paragraph.
        :return: list with the paragraphs names, list with the proportions for
            each author for each paragraph.
        :rtype: list[str], list[dict(str: float)]
        """
        # Init variables
        author_names = self.authors
        prop_authors_paragraphs = []
        paragraph_names = []
        i = 1
        for paragraph in self.paragraphs:
            # Initialize a dictionary containing participation proportion for
            # each authors
            prop_authors = {author_name: 0 for author_name in author_names}
            # Only take into account the real paragraphs (not the new lines)
            if not paragraph.new_line:
                # Create the label of the paragraph
                paragraph_names.append('p' + str(i))
                i += 1
                for op in paragraph.operations:
                    # increment with the corresponding prop
                    prop_authors[op.author] += abs(
                        op.context['proportion_paragraph']
                    )
                prop_authors_paragraphs.append(prop_authors)
        return paragraph_names, prop_authors_paragraphs

    def alternating_score(self):
        """
        Compute the alternating score that is the number of main author
        alternations between paragraphs divided by the total number of
        alternations of paragraphs.
        :return: the alternating score which is a float between 0 and 1.
            Return 0 if there is less than 2 paragraph
        """
        num_alt = 0
        main_authors = []

        # Compute the proportions for each paragraphs for each authors
        _, prop_authors_paragraphs = self.prop_paragraphs()
        for para in prop_authors_paragraphs:
            # Add the author who participated the most in the paragraph
            main_authors.append(max(para.keys(), key=(lambda key: para[key])))

        # Increment the alternation counter only when we change authors
        for i, author in enumerate(main_authors):
            if i > 0 and author != main_authors[i - 1]:
                num_alt += 1
        # Divide overall counter of alternations by maximum number of alternations
        if len(main_authors) > 1:
            return num_alt / (len(main_authors) - 1)
        # If there is only one paragraph there is no alternation
        else:
            return 0

    def user_participation_paragraph_score(self):
        """
        Compute the score of user participation per paragraph.
        :return: Score between 0 and 1 being the weighted average
            (paragraph lengths) of the proportion entropy of users
        """
        paragraph_participations = []
        _, prop_authors_paragraphs = self.prop_paragraphs()
        paragraph_lengths = []

        # Get the length of each paragraphs
        for para in self.paragraphs:
            if not para.new_line:
                paragraph_lengths.append(para.get_abs_length())

        # Compute the entropy of author proportions for each paragraphs
        for para in prop_authors_paragraphs:
            author_props = list(para.values())
            paragraph_participations.append(
                self.compute_entropy_prop(author_props, len(author_props))
            )

        # Compute the weighted average according to paragraph lengths
        overall_score = sum(np.multiply(paragraph_participations,
                                        paragraph_lengths))
        if sum(paragraph_lengths) != 0:
            return overall_score / sum(paragraph_lengths)
        else:
            # If no paragraph, we choose to return a score of zero
            return 0

    def break_score(self, break_type, start_time=0):
        """
        Compute the breaking score, i.e. the score that tells whether a pad is
        written only in one time or with multiple accesses.
        :param break_type: string that is either 'short' for short breaks or
            'day' for daily ones.
        :param start_time: start time from which to consider operations
        :return: The score is the number of breaks over the whole pad divided
            by the time spent on the pad. Between 0 and 1.
        """
        # Compute the time spent in s
        operations = Operation.sort_ops(self.operations)
        first_timestamp = max(operations[0].timestamp_start, start_time)
        last_timestamp = operations[len(operations) - 1].timestamp_end
        time_spent = (last_timestamp - first_timestamp) / 1000  # in s

        # Compute the number of breaks according to the type
        num_break = 0
        for op in self.operations:
            if op.timestamp_start >= start_time:
                if break_type == 'short':
                    if op.context['first_op_break']:
                        num_break += 1
                elif break_type == 'day':
                    if op.context['first_op_day']:
                        num_break += 1
        # Calculate the final score
        if time_spent >= 1:
            return num_break / time_spent
        else:
            # If less than 1s is spent on the pad, the score is zero
            return 0

    def type_overall_score(self, op_type=None, start_time=0):
        """
        Compute proportion of one type: write, delete, edit or paste over
        the whole pad.
        :param op_type: the operation type 'write', 'delete', 'edit' or 'paste'
            If None, returns all in a dictionary
        :param start_time: start time from which to consider operations
        :return: the proportion of the operation type
        """
        if op_type is None:
            all_types_count = {'write': 0,'delete': 0,'edit': 0,'paste': 0}
            op_count = 0
            # Count the number of operations
            for op in self.operations:
                if op.type != 'jump' and op.timestamp_start >= start_time:
                    op_count += 1
                    all_types_count[op.type] += 1


            # Calculate the overall proportion
            if op_count > 0:
                return {type_: type_count / op_count for
                        type_,type_count in all_types_count.items()}
            else:
                return all_types_count

        else:
            type_count = 0
            op_count = 0
            # Count the number of operations
            for op in self.operations:
                if op.type != 'jump' and op.timestamp_start >= start_time:
                    op_count += 1
                    if op.type == op_type:
                        type_count += 1

            # Calculate the overall proportion
            return type_count / op_count

    def user_type_score(self, op_type=None, start_time=0):
        """
        Compute the entropy for an operation type on all users except the
        admin. It creates a matrix of dimension (num_user x num_types)
        counting the number of different operations per user. It normalizes
        the counts according to the rows and then normalize the proportions
        according to the columns to compute a valid entropy for each entropy.
        :param op_type: the operation type 'write', 'delete', 'edit' or 'paste'
            If None, returns all in a dictionary
        :param start_time: start time from which to consider operations
        :return:the entropy score for one type over all users.
        """
        types = ['write', 'edit', 'delete', 'paste']
        users = self.authors[:]

        # Remove the admin
        if 'Etherpad_admin' in users:
            users.remove('Etherpad_admin')

        count_type = np.zeros((len(users), len(types)))
        for op in self.operations:
            if (op.type != 'jump' and
                op.author != 'Etherpad_admin' and
                op.timestamp_start >= start_time):

                user_idx = users.index(op.author)
                type_idx = types.index(op.type)
                count_type[user_idx, type_idx] += 1

        # Normalize the counter of op types per user
        total_type = count_type.sum(axis=1)[:, None]
        norm_type = np.divide(count_type,
                              total_type,
                              out=np.zeros_like(count_type),
                              where=total_type != 0)

        # Normalize the proportion of users per types
        total_user = norm_type.sum(axis=0)
        norm_user = np.divide(norm_type,
                              total_user,
                              out=np.zeros_like(norm_type),
                              where=total_user != 0)

        # Compute the entropy for all types
        type_scores = {}
        for idx, type_props in enumerate(norm_user.T):
            type_scores[types[idx]] = self.compute_entropy_prop(
                type_props, len(users))
        if op_type is None:
            return type_scores
        else:
            return type_scores[op_type]

    def get_text_length(self):
        text = self.get_text()
        for line in config.default_lines:
            text = text.replace(line+'\n', '')
            text = text.replace(line, '')

        return len(text)

    def get_all_text_added_length(self):
        count = 0
        write_count = 0
        paste_count = 0
        for operation in self.operations:
            if(operation.author=='Etherpad_admin'):
                continue
            length = operation.get_length_of_op()
            if length > 0:
                count += length
                if(operation.type != 'paste'):
                    write_count += length
                if(operation.type == 'paste'):
                    paste_count += length
        return [count,write_count,paste_count]

    def count_chars(self, op_type=None, start_time=0):
        """
        Count the number of characters added and/or deleted.
        :param op_type: the operation type 'add' or 'del'
            If None, returns both in a dictionary
        :param start_time: start time from which to consider operations
        :return:total number of chars added or deleted.
        """
        counted_chars = {"add": 0, "del": 0}
        for op in self.operations:
            if op.timestamp_start >= start_time:
                for elem_op in op.elem_ops:
                    if elem_op.operation_type == "add":
                        counted_chars["add"] += elem_op.get_length_of_op()
                    elif elem_op.operation_type == "del":
                        counted_chars["del"] -= elem_op.get_length_of_op()
        if op_type is None:
            return counted_chars
        else:
            assert op_type in ["add", "del"]
            return counted_chars[op_type]

    def paragraph_average_length(self):
        """
        Compute the average length (number of characters) per paragraph.
        :return: average paragraph length.
        :rtype: float
        """
        paragraph_lengths = [paragraph.length
            for paragraph in self.paragraphs
            if not paragraph.new_line]

        if len(paragraph_lengths):
            return np.mean(paragraph_lengths)
        else:
            return 0

    def superparagraph_average_length(self):
        """
        Compute the average length (number of characters) per superparagraph.
        :return: average superparagraph length.
        :rtype: float
        """
        superparagraph_lengths = [np.sum(
            [
                self.paragraphs[p_idx + superparagraph.start].length
                for p_idx in range(superparagraph.length)
            ])
            for superparagraph in self.superparagraphs
            if not superparagraph.new_line]

        if len(superparagraph_lengths):
            return np.mean(superparagraph_lengths)
        else:
            return 0

    def average_paragraphs_per_superparagraph(self):
        """
        Compute the average number of lines (paragraphs) per superparagraph.
        :return: average paragraphs per superparagraph.
        :rtype: float
        """
        superparagraph_lengths = [(superparagraph.length + 1) / 2
            for superparagraph in self.superparagraphs
            if not superparagraph.new_line]

        if len(superparagraph_lengths):
            return np.mean(superparagraph_lengths)
        else:
            return 0

    def pad_at_timestamp(self, timestamp_threshold):
        """
        Return the pad at a certain timestamp. It will contain all the
        operations that started before the timestamp
        :param timestamp_threshold: timestamp until which we take the operations
        :type timestamp_threshold: int
        :return: The new pad
        :rtype: Pad
        """
        new_pad = Pad(self.pad_name)
        elem_ops = []
        for elem_op in self.get_elem_ops(True):
            if elem_op.timestamp <= timestamp_threshold:
                if not elem_op.belong_to_operation in new_pad.operations:
                    new_pad.operations.append(elem_op.belong_to_operation)
                elem_ops.append(elem_op)
        pads, _, elem_ops_treated = operation_builder.build_operations_from_elem_ops(
            {self.pad_name: elem_ops},
            config.maximum_time_between_elem_ops
        )
        return pads[self.pad_name], elem_ops_treated[self.pad_name]

    def to_print(self,
        print_pad_name=True,
        print_text=False,
        print_text_colored_by_authors=False,
        print_text_colored_by_ops=False,
        print_metrics_text=True):
        """
        Return a string with the information to be printed.
        The boolean parameters indicate the parts of information to be
        included in (excluded from) the string.
        """
        text = ""
        if print_pad_name:
            text += "PAD: {}\n".format(self.pad_name)
        if print_text:
            text += "TEXT:\n{}\n".format(self.get_text())
        if print_text_colored_by_authors:
            text += "\nCOLORED TEXT BY AUTHOR:\n{}\n".format(
                self.display_text_colored_by_authors())
        if print_text_colored_by_ops:
            text += "\nCOLORED TEXT BY OPS:\n{}\n".format(
                self.display_text_colored_by_ops()
            )
        if print_metrics_text:
            text += "\nSCORES:\n{}".format(self.get_metrics_text())

        return text

    def get_timestamps(self):
        """
        Returns a tuple containing the minimum and maximum timestamps
        contained in the current Pad.
        """
        times = [int(eo.timestamp) for eo in self.get_elem_ops(sorted_=True)]
        if len(times):
            return (np.min(times), np.max(times))
        else:
            return (None, None)

    def get_text_by_author(self,author):
        '''
        recover the text wrote by author
        :param author:
        :return: author's text
        '''
        text = ''
        author_operation = self.author_operation[author]
        for op in author_operation:
            if op.text != '\n':
                text += op.text + ' '
        return text


    ###############################
    # Managing WindowOperation
    ###############################
    def PreprocessOperationByAuthor(self,compute_vector=False,model=None):
        '''
        group operations by author and compute distance or similarity of different authors' text
        :param compute_vector: whether to compute vector
        :param model:  the pretrained model
        :return: None
        '''
        for op in self.operations:
            if op.author=="Etherpad_admin":
                continue # ignore the Etherpad_admin edit
            if op.author not in self.author_operation.keys():
                self.author_operation[op.author] = [op]
            else:
                self.author_operation[op.author].append(op)

            if op.author not in self.author_timeStamp.keys():
                self.author_timeStamp[op.author] = [[op.timestamp_start, op.timestamp_end]]
            else:
                self.author_timeStamp[op.author].append([op.timestamp_start, op.timestamp_end])

        pad_text =' '.join(cleanText(self.get_text())) # compute the sec2vec need to clean text at first
        pad_vec = model.embed_sentence(pad_text)
        self.author_vectors["pad"] = pad_vec
        self.author_text["pad"] = pad_text
        if compute_vector==True:
            authorNum=0
            for author in self.authors:
                if author!="Etherpad_admin":
                    authorNum+=1
                    author_text = ' '.join(cleanText(self.get_text_by_author(author)))
                    author_vec = model.embed_sentence(author_text)
                    self.author_vectors[author] = author_vec
                    self.author_text[authorNum] = author_text
            AuthorList = list(self.author_vectors.keys())

            for i in range(len(AuthorList)-1):
                for j in range(i+1,len(AuthorList)):
                    author1 = AuthorList[i]
                    author2  = AuthorList[j]
                    if AuthorList[i]=='pad':
                        Author1 = 'pad' # pad is the total text
                        Author2 = str(j)
                    elif AuthorList[j]=='pad':
                        Author2 = 'pad'
                        Author1 = str(i)
                    else:
                        Author2 = str(j)
                        Author1 = str(i)
                    # compute the distance and similarity between different authors
                    self.author_distance[Author1+' VS '+Author2] = np.linalg.norm(
                     self.author_vectors[author1]- self.author_vectors[author2])

                    self.author_similarity[Author1 + ' VS ' + Author2] = 1 - spatial.distance.cosine(self.author_vectors[author1],
                                                              self.author_vectors[author2])

    def PlotLengthOperationTime(self):
        ## plot the different operations By different author on same time period .
        timeLengActor = {}
        for op in self.operations:
            if op.author not in timeLengActor.keys():
                timeLengActor[op.author]=[[],[]]

            timeLengActor[op.author][0].append((op.timestamp_end+op.timestamp_start)/2)
            for elem_op in op.elem_ops:
                L = 0
                if elem_op.operation_type=="add":
                    L += len(elem_op.text_to_add)
                else:
                    L+=elem_op.length_to_delete
            timeLengActor[op.author][1].append(L)

        jet = plt.get_cmap('coolwarm')
        for author in timeLengActor.keys():
            plt.scatter(timeLengActor[author][0],timeLengActor[author][1],cmap=jet)
        plt.show()

    def RejectOutliers(self,data, m=2):
        '''
        this is used to remove outliers points
        :param m: parameter to control outliers
        :return: list after removing outliers and remained index
        '''
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        return data[abs(data - np.mean(data)) < m * np.std(data)], np.nonzero(abs(data - np.mean(data)) < m * np.std(data))

    def PlotSimilarityDistribution(self,savefig):
        '''
        Plot windows' similarity distribution and compute fitting slope
        :para savefig: bool, whether to save figures.
        :return:
        '''
        def f_1(x, A, B):
            return A * x + B

        def f_2(x, A, B, C):
            return A * x * x + B * x + C
        k = 0
        fig, ax = plt.subplots(nrows=1, ncols=1)
        flag = False
        for author_pair in self.similarity.keys():
            # author_pair is each two authors
            similarity_dict = self.similarity[author_pair]
            if sum(list(similarity_dict.values()))==0:
                # no similarity
                continue
            else:
                ydata = list(similarity_dict.values())
                if len(ydata)>=5:
                    #ydata, indexes = self.RejectOutliers(list(similarity_dict.values()))
                    #xdata = np.array(list(similarity_dict.keys()))[indexes]
                    ydata = np.nan_to_num(np.array(ydata))
                    xdata = np.array(list(similarity_dict.keys()))
                    ax.plot(xdata,ydata,label='curve')
                    ax.scatter(xdata,ydata)
                    popt1, pcov1 = optimize.curve_fit(f_1, xdata, ydata)
                    self.fitting_slope = round(popt1[0],3)
                    ax.plot(xdata, f_1(xdata, *popt1), label='linear fit with ' + 'k='+str(round(popt1[0],3)))
                    ax.set_ylim(0.0, 1.0)
                    ax.set_xlabel('WindowOperation group')
                    ax.set_ylabel('Similarity')
                    ax.legend()
                # if xdata.size>=3:
                #     popt2, pcov2 = optimize.curve_fit(f_2, xdata, ydata)
                # #ax.plot(xdata,ydata,label = author_pair)
                #     ax.plot(xdata, f_2(xdata, *popt2), label='fit degree 2' + author_pair)
                #     ax.set_ylim(0.0, 1.0)
                #     ax.legend()
                    flag = True
        # plt.show()
        if flag and savefig:
            # if we have similarity then plot it.
            fig.savefig('../similarity_dis_img/'+self.pad_name+'.png')
            plt.close(fig)

    def selectTimeInterval(self,
	time_interval_list,
	valid_window_threshold,
	group_num_thresold):
        '''
        find the best time interval
        :param time_interval_list:  list of time interval
        :param valid_window_threshold: lower bound for number of words in a window
        :param group_num_thresold: lower bound for number of window groups in a pad
        :return: list of valid winndow percentage
        '''
        label = []
        for time_interval in time_interval_list:
            self.BuildWindowOperation(time_interval)
            self.getTextByWin(False,None)
            valid_num = 0

            for group_num in self.window_operation.keys():
                flag = True
                if len(self.window_operation[group_num])<2:
                    continue
                for winOp in self.window_operation[group_num]:
                    if winOp.text_added_len <valid_window_threshold:
                        flag = False
                        break
                if flag:
                    valid_num +=1
            if valid_num<group_num_thresold:
                label.append(0)
            else:
                label.append(1)
        return label


    def BuildWindowOperation(self,time_interval=100000):
        '''
        build windows of the whole pad
        :param time_interval:  setting the time interval of
        :return: None
        '''
        i = 1
        self.window_operation={}
        for op in self.operations:
            if op.author=='Etherpad_admin':
                continue
            difference_time = op.timestamp_end-self.start_time
            while(difference_time>time_interval*i):
                ## last window is finished then we need to sort the operation
                if self.window_operation.keys() and i in self.window_operation.keys():
                    for win in self.window_operation[i]:
                        win.generateElemOps()
                    self.window_operation[i].sort(key=self.windowSort)
                i +=1

            tmp_window_operation = WindowOperation(i,op.author,[op],time_interval)
            if i not in self.window_operation.keys(): ## the
                self.window_operation[i] = [tmp_window_operation]

            elif tmp_window_operation in self.window_operation[i]:
                self.window_operation[i][self.window_operation[i].index(tmp_window_operation)].addOperation(op)
            else:
                self.window_operation[i].append(tmp_window_operation)

    def getTextByWin(self,vector_flag,model):
        '''
        recover each window's text
        :param model:  pretrained model
        "param vector_flag: whether to compute vector
        :return:
        '''
        for group_num in self.window_operation.keys():
            for winOp in self.window_operation[group_num]:
                winOp.createWindowText(vector_flag,model)

    def computeDistance(self):
        '''
        compute the distance between different authors
        :return: None
        '''

        for groupNum in self.window_operation.keys():

            if len(self.window_operation[groupNum])<2:
                continue
            else:
                index_list = [i for i in range(len(self.window_operation[groupNum]))]
                tuples = list(combinations(index_list,2))
                for ind,tuple in enumerate(tuples):
                    index1 = tuple[0]
                    index2 = tuple[1]
                    win1 = self.window_operation[groupNum][index1]
                    win2 = self.window_operation[groupNum][index2]
                    text1 = win1.text
                    text2 = win2.text
                    author1 = win1.author
                    author2 = win2.author
                    dis = round(np.linalg.norm(win1.textVector - win2.textVector),3)
                    similarity = 1-spatial.distance.cosine(win1.textVector, win2.textVector)
                    # if dis<0.2:
                    #     dis=0
                    if math.isnan(similarity):
                        similarity = 0
                    authors = [author1,author2]
                    authors.sort()
                    key = '--VS--'.join(authors)
                    if key not in self.distance.keys():
                        self.distance[key]= {}
                        self.distance[key][groupNum] = dis
                        self.similarity[key]={}
                        self.similarity[key][groupNum] = similarity
                        self.window_operation_text[key] = {}
                        self.window_operation_text[key][groupNum] = ["-----user1:-----" + text1,
                                                           "-----user2-----" + text2]
                    else:
                        self.distance[key][groupNum] = dis
                        self.similarity[key][groupNum] = similarity
                        self.window_operation_text[key][groupNum] = ["-----user1:-----" + text1,
                                                                   "-----user2-----" + text2]

    def windowSort(self,win):
        '''
        sort function which is used to sort windows
        :param win:
        :return:  start time of window
        '''
        return win.start_time

    def operationSort(self,op):
        return op.timestamp_start

    def text_by_ops(self):
        text = ''
        for op in self.operations:
            op.getOpText()
            leng = len(text)
            text = text + op.text
            # text = text[:op.position_start_of_op] + op.text + text[op.position_start_of_op:]
        self.text = text

    def merge(self,d1, d2, merge_fn=lambda x, y: y):
        """
        Merges two dictionaries, non-destructively, combining
        values on duplicate keys as defined by the optional merge
        function.  The default behavior replaces the values in d1
        with corresponding values in d2.  (There is no other generally
        applicable merge strategy, but often you'll have homogeneous
        types in your dicts, so specifying a merge technique can be
        valuable.)
        Examples:
        >>> d1
        {'a': 1, 'c': 3, 'b': 2}
        >>> merge(d1, d1)
        {'a': 1, 'c': 3, 'b': 2}
        >>> merge(d1, d1, lambda x,y: x+y)
        {'a': 2, 'c': 6, 'b': 4}
        """
        result = dict(d1)
        for k, v in d2.items():
            if k in result:
                result[k] = merge_fn(result[k], v)
            else:
                result[k] = v
        return result

    def SortDict(self,dict):
        keys = list(dict.keys())
        keys.sort()
        values = [dict[key] for key in keys]
        return keys,values

    def PlotTextAdded(self):
        '''
        plot the length of added text according to author
        :return: None
        '''
        author_text_length = {}
        for group,wins in self.window_operation.items():
            for win in wins:
                if win.author not in author_text_length.keys():
                    author_text_length[win.author] = {group:win.text_added_len}
                else:
                    author_text_length[win.author][group]=win.text_added_len

        #Sum = {}
        colors = iter(cm.rainbow(np.linspace(0, 1, len(author_text_length.keys())+1)))
        for author in author_text_length.keys():
            #Sum = self.merge(Sum,author_text_length[author], merge_fn=lambda x, y: x+y)
            plt.scatter(author_text_length[author].keys(),author_text_length[author].values(),color=next(colors),label=author)
        #sum_keys,sum_values = self.SortDict(Sum)
        #plt.plot(sum_keys, sum_values, color=next(colors),label='Total added length')
        plt.legend()
        plt.show()
