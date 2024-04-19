import omnifilled as omf
import importlib.util

# def checklib(libname):

#     spec = importlib.util.find_spec(libname)
#     if spec is None:
#         return False
#     else:
#         return True

def main():
    # libs= ["numpy", "matplotlib", "awkward","os","tensorflow","uproot","keras","parse", "aiohttp", "coffea"]
    # unins = []
    
    # print("Checking libraries:")
    # for libname in libs:
    #     if checklib(libname):
    #         print(f"{libname}: Installed")
    #     else:
    #         unins.append(libname)
    #         print(f"{libname}: Not installed")            
    # if len(unins)>0:
    #     print("Please pip install the missing libraries:")
    #     print(unins)
    # else:
    omf.unfold() # Can pass in (testgen, testreco, trainreco) but not necessary if .root is already as desired 

if __name__ == "__main__":
    main()

