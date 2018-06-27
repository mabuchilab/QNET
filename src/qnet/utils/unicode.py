"""Utils for working with unicode strings"""
from uniseg.graphemecluster import grapheme_clusters

__all__ = []
__private__ = ['grapheme_len', 'ljust', 'rjust']


def grapheme_len(text):
    """Number of graphemes in `text`

    This is the length of the `text` when printed::
        >>> s = 'Â'
        >>> len(s)
        2
        >>> grapheme_len(s)
        1
    """
    return len(list(grapheme_clusters(text)))


def ljust(text, width, fillchar=' '):
    """Left-justify text to a total of `width`

    The `width` is based on graphemes::

        >>> s = 'Â'
        >>> s.ljust(2)
        'Â'
        >>> ljust(s, 2)
        'Â '
    """
    len_text = grapheme_len(text)
    return text + fillchar * (width - len_text)


def rjust(text, width, fillchar=' '):
    """Right-justify text for a total of `width` graphemes

    The `width` is based on graphemes::

        >>> s = 'Â'
        >>> s.rjust(2)
        'Â'
        >>> rjust(s, 2)
        ' Â'
    """
    len_text = grapheme_len(text)
    return fillchar * (width - len_text) + text
