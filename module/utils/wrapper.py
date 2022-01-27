import time
from datetime import datetime, timedelta

from cv2 import error


def handle_elaped_time(n_iter):
    print("start")
    print("n_iter:", n_iter)
    error_indexes = []
    def wrapper(func):
        print("wrapper")
        def innerfunc():
            print("innderfunc")
            elapsed_time_all = 0
            for i in range(n_iter):
                print(f"---index: {i}---")
                print("started at", datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                start = time.time()
                try:
                    func(i)
                except KeyboardInterrupt:
                    break
                except:
                    print("--------------------")
                    print("---An Error Occured!---")
                    print("--------------------")
                    print("---------At---------")
                    print("--------------------")
                    print(f"---index: {i}---")
                    print("--------------------")

                    import traceback
                    traceback.print_exc()
                    error_indexes.append(i)
                
                finally:
                    elapsed_time = time.time() - start
                    elapsed_time_all += elapsed_time
                    
                    print ("\nETA: {0}".format(elapsed_time) + "[sec]")
                    
                    finished_at = datetime.now()
                    finished_at += timedelta(seconds=elapsed_time_all * (n_iter-i)/(i+1))

                    print ("expected to be fished at", finished_at.strftime('%Y/%m/%d %H:%M:%S'))
            print("---done---")
            print("error_indexes:", error_indexes)
        return innerfunc
    return wrapper
    