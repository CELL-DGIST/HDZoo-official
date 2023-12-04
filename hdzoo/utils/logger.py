"""
HD Zoo - Yeseong Kim (CELL) @ DGIST, 2023
"""

""" Simple Logger """
class Logger:
    def __init__(self, filename, verbose=True):
        self.f = open(filename, "w")
        self.verbose = verbose

    def __del__(self):
        self.f.close()

    def i(self, *args): 
        for i, arg in enumerate(args):
            if i > 0:
                self.f.write(", ")
            self.f.write(str(arg) + "")
        self.f.write('\n')

    def d(self, *args):
        self.i(*args)
        if self.verbose:
            print(*args)


""" Set Global Similarity Metric """
def setup_global_logger(logfile):
    log.logger = Logger(logfile)


""" class to keep the global logger instance """
class log:
    logger = None

    @staticmethod
    def d(*args):
        log.logger.d(*args)

    @staticmethod
    def i(self, *args): 
        log.logger.i(*args)
