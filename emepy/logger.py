import logging
import warnings
import os 

class Logger(logging.Logger):

    def __init__(self, print_screen=True, print_file=True, print_file_name="output.log", enable_warnings=True, *args, **kwargs):
        super().__init__("logger", *args, **kwargs)
        self.pf, self.ps, self.pfn = (print_file, print_screen, print_file_name)
        self.enable_warnings = enable_warnings

        # Set up logger
        log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
        if self.pf:
            file_handler = logging.FileHandler(self.pfn)
            file_handler.setFormatter(log_formatter)
            self.addHandler(file_handler)
        if self.ps:
            screen_handler = logging.StreamHandler()
            screen_handler.setFormatter(log_formatter)
            self.addHandler(screen_handler)

    def warn(self, *args, **kwargs):
        if self.enable_warnings:
            warnings.warn(*args, **kwargs)
    
    def delete_log(self):
        if self.pf and os.path.exists(self.pfn):
            os.remove(self.pfn)

    def input(self, *args, **kwargs):
        Logger(True, self.pf, self.pfn).info(*args, **kwargs)
        return input(*args, **kwargs)