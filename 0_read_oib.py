from module import *

if __name__ == "__main__":
    import glob
    files = glob.glob("./resource/**/*.oib")

    print(files)

    @handle_elaped_time(len(files))
    def convert_oib_to_data(i):
        print(f"---filename: {files[i]}---")
        oib = OIBFile(files[i])
        oib.convert_and_export()

    convert_oib_to_data()
    