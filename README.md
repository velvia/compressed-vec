## Compressed-Vec

This is a *compressed vec* library, rather than a *compression* library.  What does that mean?

A compression library takes some uncompressed data and provides essentially compress() and decompress() functions.  Typically you have to decompress data to be able to do anything with it.

On the other hand, a *compressed vec* library gives you a `Vec` (or list, or array) like interface for working with compressed representations of specific kinds of data (in this case, integers, floats, strings, and other common entities).  This allows you to efficiently store large quantities of certain data in memory while being able to read and access it.