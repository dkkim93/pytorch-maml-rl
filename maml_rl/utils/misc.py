import logging
import git


#################################################################################
# LOGGING
#################################################################################
def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args, path="."):
    log = {}                                                                                                                                        
    set_logger(
        logger_name=args.log_name, 
        log_file=r'{0}/{1}'.format(args.output_folder, args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    repo = git.Repo(path)
    log[args.log_name].info("Branch: {}".format(repo.active_branch))
    log[args.log_name].info("Commit: {}".format(repo.head.commit))

    return log
