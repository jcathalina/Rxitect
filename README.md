# Known installation issues
### WSL2
If you are developing Rxitect using a WSL2 instance, then you might have to install pytorch in a slightly different way:
You'll have to download pytorch using pip with the correct url found in [TODO](TODO).
Here's an example using python 3.10 and cuda 11+ `pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp310-cp310-linux_x86_64.whl`. It might be the case that you have to install pytorch lightning separately after this again, but this can be done using conda in the same way as before.