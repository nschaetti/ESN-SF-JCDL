#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import re


# Remove _
def remove_underline(text):
    """
    Remove _
    :param text:
    :return:
    """
    return text.replace(u"_", u"")
# end remove_underline

# Remove - \n
def remove_line_breaks(text):
    """
    Remove - \n
    :param text:
    :return:
    """
    return text.replace(u"- \n", u"")
# end remove_line_breaks

# Remove multiple line
def remove_multiple_line(text):
    """
    Remove multiple line
    :param text:
    :return:
    """
    return text.replace(u"\n\n", u"\n")
# end remove_multiple_line

# Remove pagination
def remove_pagination(text):
    """
    Remove pagination
    :param text:
    :return:
    """
    #print(re.findall(r"\n\n\n\d+ \n\n\n", text))
    return re.sub(r"\n\n\d+ \n\n\n", u"", text)
# end remove_pagination

# Remove magazine title
def remove_magazine_title(text):
    """
    Remove magazine title
    :param text:
    :return:
    """
    text = re.sub(r"\n\nIF \n\n", u"", text)
    return re.sub(r"\n\nGALAXY \n\n", u"", text)
# end remove_magazine_title
