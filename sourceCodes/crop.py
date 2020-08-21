import utilityFunc


utilityFunc.resizeAndSave(r"C:\\Users\\Mehmet\\Desktop\\HighResPyTorch\\orijinalResimler\\label\\test_label\\*.png", r"C:\\Users\\Mehmet\\Desktop\\HighResPyTorch\\b\\", 512)

utilityFunc.crop2(r"C:\\Users\\Mehmet\\Desktop\\HighResPyTorch\\b\\*.png", r"C:\Users\Mehmet\Desktop\HighResPyTorch\\croppedDataset\\", 64, 8)

utilityFunc.savingLR(r"C:\\Users\\Mehmet\Desktop\\HighResPyTorch\\croppedDataset\\*.png", r"C:\Users\Mehmet\Desktop\HighResPyTorch\SONdata\\", 3)
