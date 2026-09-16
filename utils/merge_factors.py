rootPath = "D:\\GraduationDesign\\AP_calibration\\OUTPUT_new_scaler\\001_ord_inference_20240702_232358\\para_results\\test\\"

def merge_factors(epoch, batch_size, generate_num):
    fileDir = rootPath + "epoch" + str(epoch) + "\\factors\\0\\"
    # fileDir = rootPath + "epoch" + str(epoch) + "\\factors\\"
    # i代表当前的生成次数，j代表对应哪一个cond
    line_matrix = [[] for _ in range(batch_size)]
    for i in range(generate_num):
        filePath = fileDir + str(i) + "\\" + "e" + str(epoch).zfill(10) + ".txt"
        file = open(filePath, "r")

        for j in range(batch_size):
            line = file.readline().strip()
            line_matrix[j].append(line)
        file.close()

    
    for i in range(batch_size):
        outputFilePath = fileDir + "factors" + str(i) + ".txt"
        with open(outputFilePath, "w") as outputFile:
            for line in line_matrix[i]:
                outputFile.write(line)
                outputFile.write("\n")


if __name__ == "__main__":
    epochs = [-1]
    batch_size = 12
    generate_num = 100
    for epoch in epochs:
        merge_factors(epoch, batch_size, generate_num)