// registration of retinal OCTA images based on supreficial image
// Developed by Team Yanagi (2025)
// For Ariake Study only
// version 2.0 (2025/12/1)

#@ File (label = "Input directory", style = "directory") mainDir
#@ File (label = "Results Output directory", style = "directory") outputDir
#@ boolean (label = "Apply CLAHE to Stack 1 (Reference)", value = false) applyClaheToImage1

mainDir = mainDir + File.separator;
outputDir = outputDir + File.separator;


// グローバル変数：バッチモードの状態を追跡
var isBatchModeActive = false;
var qcResults = newArray();
var startTime = getTime();
var DEBUG = false;
var skip = 0;


// ログウィンドウをクリアして開始
print("\\Clear");
print("=================================================");
print("ARIAKE Stack Registration v2.1");
print("=================================================");
print("Input directory: " + mainDir);
print("Output directory: " + outputDir);
print("Image scale factor: 4x");
print("Default transformation: Affine");

claheStatus = "No";
if (applyClaheToImage1) {
    claheStatus = "Yes";
}
print("CLAHE to Stack 1: " + claheStatus);
print("================================================="); 

do {
	mainFolder = File.getName(mainDir);

	print("\n--- Processing folder: " + mainFolder + " ---");
	print("Timestamp: " + getDateTimeString());

// 出力ディレクトリのサブフォルダ作成
outputSubDir = outputDir + mainFolder + File.separator;
if (!File.exists(outputSubDir)) {
    File.makeDirectory(outputSubDir);
    print("Created output subdirectory: " + outputSubDir);
    print("  Timestamp: " + getDateTimeString());
}

// 処理開始前に一度全ての画像を閉じる
run("Close All"); 

// サブディレクトリのリスト取得
listSubDirs = getFileList(mainDir);


// 各サブディレクトリ (visit) で処理を実行
for (i = 0; i < listSubDirs.length; i++) {
    subDirPath = mainDir + listSubDirs[i];
    listSubSubDirs = getFileList(subDirPath);

    if (endsWith(listSubDirs[i], "/")) {
        listSubDirs[i] = replace(listSubDirs[i], "/", "");
    }

    stackname = listSubDirs[i];
    print("\n[" + (i+1) + "/" + listSubDirs.length + "] Processing: " + stackname);
    print("  Start time: " + getDateTimeString());
    
    var visitStartTime = getTime();
    
    // --- 1. スタック作成 ---
    enableBatchMode();
    print("  >> Step 1: Creating stacks...");
    skip =0;
    stack_formation_new(subDirPath, stackname, skip); 
    print("  << Step 1 completed at: " + getDateTimeString());
    
    // --- 2. 登録処理 ---
    print("  >> Step 2: Starting registration...");
    multi_stack_reg();
    print("  << Step 2 completed at: " + getDateTimeString());
    
    var visitEndTime = getTime();
    var visitElapsed = (visitEndTime - visitStartTime) / 1000;
    print("  Visit processing time: " + d2s(visitElapsed, 2) + " seconds");
}

// show_registered_images (confirmation)
image_folder = outputSubDir;
print("\n--- Displaying registered images for confirmation ---");
disableBatchMode();
folder_img_selection(image_folder);

// ユーザーに結果を確認
	Dialog.create("確認");
        Dialog.addMessage("他の画像に進みますか?");
        Dialog.addChoice("選択:はいの場合には次に患者フォルダを指定", newArray("はい", "いいえ、終了する"));
        Dialog.show();
        
        userChoice = Dialog.getChoice();
                
if (userChoice == "はい") { 
    run("Close All");
    mainDir = getDirectory("Choose the main directory");
    print("\n=================================================");
    print("Next patient folder selected: " + mainDir);
    print("Timestamp: " + getDateTimeString());
    print("=================================================");
    }
} while (userChoice == "はい");

// 終了時刻とログ保存
endTime = getTime();
endDateTime = getDateTimeString();
elapsedTime = (endTime - startTime) / 1000; // 秒単位

print("\n=================================================");
print("ARIAKE Stack Registration completed successfully!");
print("End time: " + endDateTime);
print("Total processing time: " + d2s(elapsedTime, 2) + " seconds");
print("Version: 2.1");
print("=================================================");

// ログをファイルに保存
saveLogToFile();

///////////////////////////////////////////////////////////////////////////////
// 日時文字列を取得する関数
///////////////////////////////////////////////////////////////////////////////
function getDateTimeString() {
    getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
    month = month + 1; // 月は0から始まるので+1
    
    monthStr = d2s(month, 0);
    if (month < 10) monthStr = "0" + monthStr;
    
    dayStr = d2s(dayOfMonth, 0);
    if (dayOfMonth < 10) dayStr = "0" + dayStr;
    
    hourStr = d2s(hour, 0);
    if (hour < 10) hourStr = "0" + hourStr;
    
    minStr = d2s(minute, 0);
    if (minute < 10) minStr = "0" + minStr;
    
    secStr = d2s(second, 0);
    if (second < 10) secStr = "0" + secStr;
    
    return d2s(year, 0) + "-" + monthStr + "-" + dayStr + " " + hourStr + ":" + minStr + ":" + secStr;
}


///////////////////////////////////////////////////////////////////////////////
// ログをファイルに保存する関数
///////////////////////////////////////////////////////////////////////////////
function saveLogToFile() {
    logContent = getInfo("log");
    
    if (lengthOf(logContent) == 0) {
        print("Warning: No log content to save.");
        return;
    }
    
    dateTimeStr = getDateTimeString();
    logFileName = "ARIAKE_Registration_Log_v2.1_" + dateTimeStr + ".txt";
    logFilePath = outputDir + mainFolder + "_log" + File.separator;
 
 if (!File.exists(logFilePath)) {
    File.makeDirectory(logFilePath);
    print("Created output subdirectory: " + logFilePath);
    print("  Timestamp: " + getDateTimeString());
}

    logFilePath = outputDir + mainFolder + "_log" + File.separator + logFileName;

	// Add header information to log
    logHeader = "=================================================\n";
    logHeader += "ARIAKE Stack Registration v2.1\n";
    logHeader += "Log File Generated: " + getDateTimeString() + "\n";
    logHeader += "=================================================\n\n";
    
    fullLog = logHeader + logContent;
    
    File.saveString(fullLog, logFilePath);
    
    print("\n=================================================");
    print("Log file saved successfully");
    print("File: " + logFileName);
    print("Path: " + logFilePath);
    print("=================================================");
    showMessage("Log Saved", "Log file has been saved to:\n" + logFilePath);
}


///////////////////////////////////////////////////////////////////////////////
// 画像を8bitに変換し、サイズ(ppi)を4倍に拡大する関数
///////////////////////////////////////////////////////////////////////////////
function quadrupleimgsize(){
	run("8-bit");
	imageWidth = getWidth(); 
	imageHight = getHeight();
	run("Size...", "width=" + imageWidth*4 + " height=" + imageHight*4 + " constrain average interpolation=Bicubic");
}

///////////////////////////////////////////////////////////////////////////////
// 開いているスタック画像のマルチスタック登録(レジストレーション)を実行
///////////////////////////////////////////////////////////////////////////////
function multi_stack_reg(){
    enableBatchMode();
    
    regStartTime = getTime();
    print("  Starting multi-stack registration...");
    print("    Timestamp: " + getDateTimeString());
    
    listAllTitles = getList("image.titles");
    actualNumImages = listAllTitles.length;
    
    print("    Total open images: " + actualNumImages);
    
    // スタック画像のリストを取得
    listStackTitles = newArray(); 
    listStackIDs = newArray(); 
    
    for (idx = 0; idx < actualNumImages; idx++) {
        selectWindow(listAllTitles[idx]); 
        currentTitle = getTitle();
        currentID = getImageID(); 
        currentSlices = nSlices;
        if (currentSlices > 1) {
            listStackTitles = Array.concat(listStackTitles, currentTitle);
            listStackIDs = Array.concat(listStackIDs, currentID);
            if (DEBUG)print("      Found stack: " + currentTitle + " (" + currentSlices + " slices)");
        }
    }
    
    if (listStackTitles.length == 0) {
        disableBatchMode();
        print("    ERROR: No stack images found");
        showMessage("Error", "No stack images found. Please open stack images first.");
        return;
    }
    
    if (listStackTitles.length == 1) {
        disableBatchMode();
        print("    INFO: Only one stack found, registration skipped");
        showMessage("Info", "Only one stack image found. Registration skipped.");
        return;
    }
    
    print("    Found " + listStackTitles.length + " stack images for registration");
    
    baseTitle = listStackTitles[4];
    selectWindow(baseTitle);
    print("    Using combined stack as reference for alignment: " + baseTitle);
    
    // Phase 1: 変換行列の計算（位置合わせ）
    tempTransformationFile = getDirectory("temp") + "transformation_" + getTime() + ".txt";
    print("    Phase 1/3: Calculating transformation parameters (based on " + baseTitle + ")...");
    
            run("MultiStackReg", "stack_1=[" + baseTitle + "] " +
        "action_1=Align " +
        "file_1=[" + tempTransformationFile + "] " +
        "stack_2=None action_2=Ignore file_2=[] " +
        "transformation=Affine save");
    
    if (!File.exists(tempTransformationFile)) {
        disableBatchMode();
        print("    ERROR: Transformation file could not be created at: " + tempTransformationFile);
        print("    Timestamp: " + getDateTimeString());
        showMessage("Error", "Transformation file could not be created. Check MultiStackReg plugin and directory permissions.");
        return;
    }
    print("    Transformation parameters calculated successfully");
    print("    Transformation file: " + tempTransformationFile);
    
    // Phase 2: 変換情報をすべてのスタックに適用 (Stack 1を含む)
    phase2StartTime = getTime();
    stacks_to_be_registered = listStackTitles.length - 1;
    print("    Phase 2/3: Applying calculated transformations to ALL stacks (1-" + stacks_to_be_registered + ")...");
    for (i = 0; i < stacks_to_be_registered; i++) {
        selectWindow(listStackTitles[i]);
        print("      [" + (i+1) + "/" + stacks_to_be_registered + "] Transforming: " + listStackTitles[i]);
        
        run("MultiStackReg", "stack_1=[" + listStackTitles[i] + "] " +
            "action_1=[Load Transformation File] " +
            "file_1=[" + tempTransformationFile + "] " +
            "stack_2=None action_2=Ignore file_2=[] " +
            "transformation=Affine");
    }
    
    phase2EndTime = getTime();
    phase2Elapsed = (phase2EndTime - phase2StartTime) / 1000;
    print("    Phase 2 completed in " + d2s(phase2Elapsed, 2) + " seconds");
    
    // Phase 3: CLAHE処理の適用
    phase3StartTime = getTime();
    claheLogStatus = "No";
    if (applyClaheToImage1) {
        claheLogStatus = "Yes";
    }
    print("    Phase 3/3: Applying CLAHE processing based on user setting (Stack 1 CLAHE: " + claheLogStatus + ")...");
    
    if (applyClaheToImage1) {
        print("      Applying CLAHE to ALL registered stacks (1-" + stacks_to_be_registered + ")...");
        for (i = 0; i < stacks_to_be_registered; i++) {
            print("        [" + (i+1) + "/" + stacks_to_be_registered + "] Applying CLAHE to: " + listStackTitles[i]);
            selectWindow(listStackTitles[i]);
            applyOptimalCLAHEToStackByTitle(listStackTitles[i]); 
        }
    } else {
        print("      Applying CLAHE to registered stacks 2-" + stacks_to_be_registered + " only...");
        for (i = 1; i < stacks_to_be_registered; i++) {
            print("        [" + i + "/" + (stacks_to_be_registered-1) + "] Applying CLAHE to: " + listStackTitles[i]);
            selectWindow(listStackTitles[i]);
            applyOptimalCLAHEToStackByTitle(listStackTitles[i]); 
        }
    }
    
    phase3EndTime = getTime();
    phase3Elapsed = (phase3EndTime - phase3StartTime) / 1000;
    print("    Phase 3 completed in " + d2s(phase3Elapsed, 2) + " seconds");

    // 出力サブディレクトリの確認と作成
    if (!File.exists(outputSubDir)) {
        File.makeDirectory(outputSubDir);
        print("    Created output subdirectory: " + outputSubDir);
    }
    
    // 平均化と8ビット変換を実行して保存
    print("    Creating average projections and saving...");
    savedCount = 0;
    for (i = 0; i < stacks_to_be_registered; i++) {
        selectWindow(listStackTitles[i]);
        
        run("Z Project...", "projection=[Average Intensity]");
        avgTitle = "AVG_" + listStackTitles[i];
        rename(avgTitle);
        
        run("8-bit");
        
        outputFileName = outputSubDir + mainFolder + "-Avg-" + listStackTitles[i] + ".tif";
        saveAs("Tiff", outputFileName);
        print("      [" + (i+1) + "/" + stacks_to_be_registered + "] Saved: " + outputFileName);
        savedCount++;
        
        close();
    }
    
    print("    Total files saved: " + savedCount);
    
    // 一時ファイルの削除
    if (File.exists(tempTransformationFile)) {
        if (File.delete(tempTransformationFile)) {
            print("    Temporary transformation file deleted: " + tempTransformationFile);
        } else {
            print("    WARNING: Failed to delete temporary file: " + tempTransformationFile);
        }
    } else {
        print("    INFO: Temporary transformation file not found (already deleted or never created).");
    }
    
    run("Close All");
    
    disableBatchMode();
    
    regEndTime = getTime();
    regElapsed = (regEndTime - regStartTime) / 1000;
    print("  Multi-stack registration completed successfully!");
    print("    Total registration time: " + d2s(regElapsed, 2) + " seconds");
    print("    End timestamp: " + getDateTimeString());
}

///////////////////////////////////////////////////////////////////////////////
// CLAHE関連の関数群
///////////////////////////////////////////////////////////////////////////////

function applyOptimalCLAHEToStackByTitle(stackTitle) {
    print("        [CLAHE] Processing stack: " + stackTitle);
    
    // バッチモード中は、タイトルではなくインデックスで検索
    var targetIndex = -1;
    for (var idx = 1; idx <= nImages; idx++) {
        selectImage(idx);
        if (getTitle() == stackTitle) {
            targetIndex = idx;
            break;
        }
    }
    
    if (targetIndex == -1) {
        print("        ERROR: Could not find image with title: " + stackTitle);
        print("        Available images:");
        for (var idx = 1; idx <= nImages; idx++) {
            selectImage(idx);
            print("          " + idx + ": " + getTitle());
        }
        return;
    }
    
    selectImage(targetIndex);
    if (DEBUG) print("        [CLAHE] Found at index: " + targetIndex);
    
    print("        Finding optimal CLAHE parameters...");
    
    blocksizes = newArray(8, 16, 32);
    hist_bins_array = newArray(128, 256);  
    max_slopes = newArray(2.0, 3.0, 4.0);
    
    best_score = 0;
    best_block = 16;
    best_bins = 256;
    best_slope = 3.0;
    
    originalSlice = getSliceNumber();
    middleSlice = Math.round(nSlices / 2);
    setSlice(middleSlice);
    
    uniqueTestName = "test_slice_" + getTime(); 
    run("Duplicate...", "title=" + uniqueTestName);
    testSliceTitle = getTitle();
    
    print("        Testing " + (blocksizes.length * hist_bins_array.length * max_slopes.length) + " parameter combinations on slice " + middleSlice + "...");
    
    for (b = 0; b < blocksizes.length; b++) {
        for (h = 0; h < hist_bins_array.length; h++) {
            for (s = 0; s < max_slopes.length; s++) {
                current_block = blocksizes[b];
                current_bins = hist_bins_array[h]; 
                current_slope = max_slopes[s];
                
                score = testCLAHEParameters(testSliceTitle, current_block, current_bins, current_slope);
                
                if (score > best_score) {
                    best_score = score;
                    best_block = current_block;
                    best_bins = current_bins; 
                    best_slope = current_slope;
                }
            }
        }
    }
    
    selectWindow(testSliceTitle);
    close();
    
    print("        Optimal parameters found: blocksize=" + best_block + ", bins=" + best_bins + ", slope=" + best_slope + " (score=" + d2s(best_score, 3) + ")");
    
    // 元のスタックに戻る
    selectImage(targetIndex);
    setSlice(originalSlice); 
    
    if (bitDepth() != 8) {
        run("8-bit");
    }
    
    print("        Applying CLAHE to entire stack");
    run("Enhance Local Contrast (CLAHE)",
        "blocksize=" + best_block +
        " histogram=" + best_bins + 
        " maximum=" + best_slope +
        " mask=*None* fast_(less_accurate)");
    
    print("        CLAHE processing completed");
}

function applyOptimalCLAHEToStack(stackID) { 
    // IDではなくタイトルで画像を探す
    var targetTitle = "";
    for (var idx = 1; idx <= nImages; idx++) {
        selectImage(idx);
        if (getImageID() == stackID) {
            targetTitle = getTitle();
            break;
        }
    }
    
    if (targetTitle == "") {
        print("        ERROR: Could not find image with ID " + stackID);
        return;
    }
    
    selectWindow(targetTitle);
    originalID = getImageID(); 
    
    print("        Finding optimal CLAHE parameters...");
    
    blocksizes = newArray(8, 16, 32);
    hist_bins_array = newArray(128, 256);  
    max_slopes = newArray(2.0, 3.0, 4.0);
    
    best_score = 0;
    best_block = 16;
    best_bins = 256;
    best_slope = 3.0;
    
    originalSlice = getSliceNumber();
    middleSlice = Math.round(nSlices / 2);
    setSlice(middleSlice);
    
    uniqueTestName = "test_slice_" + stackID + "_" + getTime(); 
    run("Duplicate...", "title=" + uniqueTestName);
    testSliceTitle = getTitle();
    
    print("        Testing " + (blocksizes.length * hist_bins_array.length * max_slopes.length) + " parameter combinations on slice " + middleSlice + "...");
    
    for (b = 0; b < blocksizes.length; b++) {
        for (h = 0; h < hist_bins_array.length; h++) {
            for (s = 0; s < max_slopes.length; s++) {
                current_block = blocksizes[b];
                current_bins = hist_bins_array[h]; 
                current_slope = max_slopes[s];
                
                score = testCLAHEParameters(testSliceTitle, current_block, current_bins, current_slope);
                
                if (score > best_score) {
                    best_score = score;
                    best_block = current_block;
                    best_bins = current_bins; 
                    best_slope = current_slope;
                }
            }
        }
    }
    
    selectWindow(testSliceTitle);
    close();
    
    print("        Optimal parameters found: blocksize=" + best_block + ", bins=" + best_bins + ", slope=" + best_slope + " (score=" + d2s(best_score, 3) + ")");
    
    selectWindow(targetTitle);
    setSlice(originalSlice); 
    
    if (bitDepth() != 8) {
        run("8-bit");
    }
    
    print("        Applying CLAHE to entire stack (this may take a moment)...");
    run("Enhance Local Contrast (CLAHE)",
        "blocksize=" + best_block +
        " histogram=" + best_bins + 
        " maximum=" + best_slope +
        " mask=*None* fast_(less_accurate)");
    
    print("        CLAHE processing completed");
}

function testCLAHEParameters(sliceTitle, blocksize, bins, slope) {
    selectWindow(sliceTitle);
    
    uniqueTempName = "temp_test_" + getTime();
    run("Duplicate...", "title=" + uniqueTempName);
    
    run("Enhance Local Contrast (CLAHE)", 
        "blocksize=" + blocksize + 
        " histogram=" + bins +
        " maximum=" + slope +
        " mask=*None* fast_(less_accurate)");
    
    quality_score = calculateQuality();
    
    close();
    
    return quality_score;
}

function calculateQuality() {
    getRawStatistics(nPixels, mean, min, max, std);
    
    getHistogram(values, counts, 256);
    
    entropy = 0;
    for (i = 0; i < 256; i++) {
        if (counts[i] > 0) {
            p = counts[i] / nPixels;
            entropy = entropy - (p * log(p) / log(2));
        }
    }
    
    contrast = std / 128.0;
    
    final_score = entropy * 0.8 + contrast * 0.2;
    
    return final_score;
}

///////////////////////////////////////////////////////////////////////////////
// スタック作成関数
///////////////////////////////////////////////////////////////////////////////
function stack_formation_new(subDirPath, stackname, skip) {
    enableBatchMode();

        run("Close All");
	enableBatchMode();
        stack_formation(subDirPath, stackname, skip);

        // バッチモード中でも nImages を使って全画像にアクセス可能
        // getList("image.titles") の代わりに、画像IDを直接使用
		print("Stacking Total images (nImages): " + nImages + ": This may take time...");
        
        listStackTitles = newArray();
        listStackIDs = newArray();
 
// サブフォルダーをリストアップ
subfolders = getFileList(subDirPath);
// サブフォルダーごとに処理を実行
for (sf = 0; sf < subfolders.length; sf++) {
    currentFolder = subDirPath + subfolders[sf];
    list = getFileList(currentFolder);
    
    // 画像が2枚以上あるか確認
    if (list.length < 2) {
        // print("Error: At least two images are required in folder: " + subfolders[sf]);
        continue;
    }
    
// 最初の画像を開く
	open(currentFolder + File.separator + list[0]);
	rename("Temp_Image");
	run("8-bit");
	quadrupleimgsize();
    	width = getWidth();
        height = getHeight();
                        
// 新しいZスタックを作成 (最初のスライスを含む)
	newImage("Stack" + list[0], "8-bit black", getWidth(), getHeight(), list.length);
	selectWindow("Temp_Image");
	pretreat();
	run("Copy");
	selectWindow("Stack" + list[0]);
	Stack.setSlice(1);
	run("Paste");
	// 一時的な画像ウィンドウを閉じる
    close("Temp_Image");

// 他の画像をスタックに追加
for (i = 1; i < list.length; i++) {
    // 他の画像を開く
    open(currentFolder + File.separator + list[i]);
    quadrupleimgsize();
    rename("Temp_Image");
    run("8-bit");
	pretreat();    
    // 画像をスタックに追加
    selectWindow("Temp_Image");
    run("Copy");
    selectWindow("Stack" + list[0]);
    Stack.setSlice(i + 1);
    run("Paste");
    
    // 一時的な画像ウィンドウを閉じる
    close("Temp_Image");
}
    // 網膜のstack画像を一枚作成
	originalStack = getImageID();
	run("Z Project...", "projection=[Average Intensity]");    
	selectImage(originalStack);
	close();
}

  // ４枚作成された画像をstackに スタックを作成し、名前をリネーム stackname
        run("Images to Stack", "name=TempStack use");
        rename("Stack_" + stackname + "_image5");  // スタック名に画像番号を付与
       
        // nImagesを使って全ての画像をチェック
        for (i = 1; i <= nImages; i++) {
            selectImage(i);
            currentTitle = getTitle();
            currentID = getImageID();
            currentSlices = nSlices;
            
            print(" Checking image " + i + ": " + currentTitle + " (slices=" + currentSlices + ")");
            
            if (currentSlices > 1) {
                listStackTitles = Array.concat(listStackTitles, currentTitle);
                listStackIDs = Array.concat(listStackIDs, currentID);
                print(" found stack: " + currentTitle + " (ID:" + currentID + ")");
            }
        }

        if (lengthOf(listStackTitles) == 0) {
            print("[ERROR] No stacks found after stack_formation. Aborting visit " + visitNumber);
            return;
        }

        // デバッグ: 見つかったスタックを全て表示
        print(" Total stacks found: " + lengthOf(listStackTitles));
        for (i = 0; i < lengthOf(listStackTitles); i++) {
            print("   Stack " + (i+1) + ": " + listStackTitles[i]);
        }
		
}

function stack_formation(subDirPath, stackname, skip){    
    enableBatchMode();
    
    print("  Creating stacks for: " + stackname);
    
    subsubDirPath = subDirPath + listSubSubDirs[0];
    images = getFileList(subsubDirPath);
    
    imageFiles = newArray();
    for (k = 0; k < images.length; k++) {
        if (endsWith(images[k], ".jpg") || endsWith(images[k], ".tif")) {
            imageFiles = Array.concat(imageFiles, images[k]);
        }
    }
    
    totalImages = imageFiles.length;
    print("    Found " + totalImages + " images per folder");
    print("    Number of subfolders: " + listSubSubDirs.length);

    
    createdStackIDs = newArray();
    
    for (n = 0; n < totalImages; n++) {
        print("      Processing image set " + (n+1) + "/" + totalImages);
        
        imagesToStack = newArray();
        
        for (j = 0; j < listSubSubDirs.length; j++) {
            if ((j + 1) == skip) {
            	    print("        Skipping folder: " + listSubSubDirs[j]);
                continue;
            }
            
            subsubDirPath = subDirPath + listSubSubDirs[j];
            images = getFileList(subsubDirPath);

            imageFiles = newArray();
            for (k = 0; k < images.length; k++) {
                if (endsWith(images[k], ".jpg") || endsWith(images[k], ".tif")) {
                    imageFiles = Array.concat(imageFiles, images[k]);
                }
            }
        
            if (n < imageFiles.length) {
                open(subsubDirPath + imageFiles[n]);
                quadrupleimgsize();
                imagesToStack = Array.concat(imagesToStack, getImageID());
            } else {
                print("        Warning: Missing image " + (n+1) + " in folder: " + listSubSubDirs[j]);
                continue;
            }
        }
        
 if (DEBUG) print("         Opened " + imagesToStack.length + " images for this stack");
 if (DEBUG) print("         Total images including existing stacks: nImages = " + nImages);
        
        if (imagesToStack.length > 0) {
            run("Images to Stack", "name=TempStack"); 
            stackName = "Stack_" + stackname + "_image" + (n+1);
            rename(stackName);
            
            createdStackIDs = Array.concat(createdStackIDs, getImageID());
            
            if (DEBUG) print("         After creating " + stackName + ", total nImages = " + nImages);
            print("        Created: " + stackName);
        }
    }
    
    print("    Successfully created " + totalImages + " stacks");
    
    
}

///////////////////////////////////////////////////////////////////////////////
// 出力フォルダ内の登録済み画像を表示してユーザー確認を行う関数
///////////////////////////////////////////////////////////////////////////////

function folder_img_selection(image_folder){
do{ 
run("Close All");
list = getFileList(image_folder);

var tifFiles = newArray();

for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tif") || endsWith(list[i], ".jpg")) {
        tifFiles = Array.concat(tifFiles, list[i]);
    }
}

if (tifFiles.length == 0) {
    print("  ERROR: No TIF files found in output folder");
    return; 
}

print("  Found " + tifFiles.length + " TIF files for confirmation");

for (i = 0; i < tifFiles.length; i++) {
    open(image_folder + tifFiles[i]);
}

	run("Images to Stack", "name=file title=[] use");
	originalStack = getImageID();	

	columns = 4;
	var rows = floor((nSlices + columns - 1) / columns);

run("Make Montage...", "columns=" + columns + " rows=" + rows + " scale=1.0");
selectImage(originalStack);
close();

var visitNumbers = newArray();

if (listSubDirs.length > 0) {
    for (i = 0; i < listSubDirs.length; i++) {
        currentDirName = listSubDirs[i];
        if (endsWith(currentDirName, File.separator)) {
            currentDirName = replace(currentDirName, File.separator, "");
        }
        if (lengthOf(currentDirName) > 0) {
            visitNumbers = Array.concat(visitNumbers, currentDirName);
        }
    }
} else {
    print("Warning: listSubDirs is empty. Skipping re-registration choice.");
    run("Close All");
    return;
}

Dialog.create("Select a Visit Number");
Dialog.addCheckbox("All images are fine", true);
Dialog.addChoice("If not, select one Visit to correct", visitNumbers);
Dialog.show();

OK = Dialog.getCheckbox();
if (!OK){
    var selectedVisitNumber = Dialog.getChoice();
    print("  User requested re-registration for: " + selectedVisitNumber);
    run("Close All");
    
    // re_registration関数を呼び出し
    re_registration(selectedVisitNumber);
    
} else {
    print("  User confirmed: All images are fine");
} 
} while (!OK);
run("Close All");
}

///////////////////////////////////////////////////////////////////////////////
// re_registration 関数
///////////////////////////////////////////////////////////////////////////////
function re_registration(visitNumber) {
    print("\n=================================================");
    print("=== Re-registration started for visit: " + visitNumber + " ===");
    print("Start timestamp: " + getDateTimeString());
    print("=================================================");
    
    reregStartTime = getTime();
    
    var i, j, k;
    var allTitles, listStackTitles, listStackIDs;
    var finalTitles, finalStackTitles, finalStackIDs;
    var tempTransformationFile;
    var userChoice;
    var subDirPath, stackname;
    var avgTitle, outputFileName, safeName;
    var list, avgFiles;
    var columns, rows, originalStack;
    var images, imageFiles, subsubDirPath;
    var transformation, reference;
    var listSubSubDirs;
    var currentID, currentTitle;
    var title_i;
    
    transformation = "Affine";
    reference = "";
    skip = 0;
    subDirPath = mainDir + visitNumber + File.separator;
    stackname = visitNumber;

    tempTransformationFile = getDirectory("temp") + "transformation_" + getTime() + ".txt";

    listSubSubDirs = getFileList(subDirPath);

    do {
        print("\n===  Re-registration start for visit: " + visitNumber + " ===");

        run("Close All");
	enableBatchMode();
        
        // 新しい方式でスタック作成
 
        stack_formation_new(subDirPath, stackname, skip);

        print("Stacking Total images (nImages): " + nImages + ": This may take time ... ");
        
        listStackTitles = newArray();
        listStackIDs = newArray();

        // nImagesを使って全ての画像をチェック
        for (i = 1; i <= nImages; i++) {
            selectImage(i);
            currentTitle = getTitle();
            currentID = getImageID();
            currentSlices = nSlices;
            
            if (DEBUG) print(" Checking image " + i + ": " + currentTitle + " (slices=" + currentSlices + ")");
            
            if (currentSlices > 1) {
                listStackTitles = Array.concat(listStackTitles, currentTitle);
                listStackIDs = Array.concat(listStackIDs, currentID);
            if (DEBUG) print(" found stack: " + currentTitle + " (ID:" + currentID + ")");
            }
        }

        if (lengthOf(listStackTitles) == 0) {
            print("[ERROR] No stacks found after stack_formation. Aborting visit " + visitNumber);
            return;
        }

        // デバッグ: 見つかったスタックを全て表示
        print(" Total stacks found: " + lengthOf(listStackTitles));
        for (i = 0; i < lengthOf(listStackTitles); i++) {
            print("   Stack " + (i+1) + ": " + listStackTitles[i]);
        }

        // デフォルト値の設定
        var defaultReference = listStackTitles[lengthOf(listStackTitles)-1];
        if (reference != "" && arrayContains(listStackTitles, reference)) {
            defaultReference = reference;
        }

        Dialog.create("基準画像選択(image5を推奨)");
        Dialog.addChoice("基準画像:", listStackTitles, defaultReference);
        Dialog.addChoice("変換タイプ:", newArray("Translation","Rigid Body","Scaled Rotation","Affine"), transformation);
        Dialog.show();
        reference = Dialog.getChoice();
        transformation = Dialog.getChoice();
        print(" Selected reference=" + reference + ", transformation=" + transformation);

        print(" Applying CLAHE if requested");
        
        print(" Before CLAHE - nImages: " + nImages);
        for (var debugIdx = 1; debugIdx <= nImages; debugIdx++) {
            selectImage(debugIdx);
            print("   Image " + debugIdx + ": " + getTitle());
        }
        
        if (applyClaheToImage1) {
            for (i = 0; i < lengthOf(listStackTitles); i++) {
                print(" About to apply CLAHE to: " + listStackTitles[i]);
                applyOptimalCLAHEToStackByTitle(listStackTitles[i]);
            }
        } else {
            for (i = 1; i < lengthOf(listStackTitles); i++) {
                print(" About to apply CLAHE to: " + listStackTitles[i]);
                applyOptimalCLAHEToStackByTitle(listStackTitles[i]);
            }
        }

        print(" Running MultiStackReg Align on reference: " + reference);
        
        var refFound = false;
        for (var refIdx = 1; refIdx <= nImages; refIdx++) {
            selectImage(refIdx);
            if (getTitle() == reference) {
                refFound = true;
                print(" Reference found at index: " + refIdx);
                break;
            }
        }
        
        if (!refFound) {
            print("[ERROR] Reference window '" + reference + "' not found. Using first stack instead.");
            reference = listStackTitles[0];
            selectImage(1);
        }
        
        run("MultiStackReg", "stack_1=[" + reference + "] action_1=Align file_1=[" + tempTransformationFile + "] stack_2=None action_2=Ignore file_2=[] transformation=[" + transformation + "] save");

        print(" Applying transformation to other stacks (excluding reference)");
        for (i = 0; i < lengthOf(listStackTitles); i++) {
            title_i = listStackTitles[i];
            if (title_i == reference) {
                print(" Skipping reference stack: " + title_i);
                continue;
            }
            
            print(" Transforming: " + title_i);
            
            var stackFound = false;
            for (var stackIdx = 1; stackIdx <= nImages; stackIdx++) {
                selectImage(stackIdx);
                if (getTitle() == title_i) {
                    stackFound = true;
                    run("MultiStackReg", "stack_1=[" + title_i + "] action_1=[Load Transformation File] file_1=[" + tempTransformationFile + "] stack_2=None action_2=Ignore file_2=[] transformation=[" + transformation + "]");
                    break;
                }
            }
            
            if (!stackFound) {
                print("[WARN] Stack not found: " + title_i);
            }
        }

        if (File.exists(tempTransformationFile)) {
            File.delete(tempTransformationFile);
            print(" Deleted temp transformation file: " + tempTransformationFile);
        }

        print(" Checking for final stacks (nImages): " + nImages);
        
        finalStackTitles = newArray();
        finalStackIDs = newArray();
        
        for (i = 1; i <= nImages; i++) {
            selectImage(i);
            currentTitle = getTitle();
            currentID = getImageID();
            currentSlices = nSlices;
            
            if (currentSlices > 1) {
                finalStackTitles = Array.concat(finalStackTitles, currentTitle);
                finalStackIDs = Array.concat(finalStackIDs, currentID);
            }
        }
        
        print(" Final stacks found: " + lengthOf(finalStackTitles));

        print(" Creating average projections and saving...");
        for (i = 0; i < lengthOf(finalStackIDs)-1; i++) {
            currentID = finalStackIDs[i];
            currentTitle = finalStackTitles[i];

            selectImage(currentID);

            run("Z Project...", "projection=[Average Intensity]");

            avgTitle = getTitle();
            run("8-bit");
            
            safeName = replace(currentTitle, " ", "_");
            outputFileName = outputSubDir + mainFolder + "-Avg-" + safeName + ".tif";
            saveAs("Tiff", outputFileName);
            print(" Saved: " + outputFileName);
            
            if (isOpen(avgTitle)) {
                selectWindow(avgTitle);
                close();
            }
            if (isOpen(currentTitle)) {
                selectWindow(currentTitle);
                close();
            }
        }

        run("Close All");
        list = getFileList(outputSubDir);
        avgFiles = newArray();
        for (i = 0; i < lengthOf(list); i++) {
            if (endsWith(list[i], ".tif") && startsWith(list[i], mainFolder + "-Avg-Stack_" + visitNumber)) {
                open(outputSubDir + list[i]);
                avgFiles = Array.concat(avgFiles, list[i]);
            }
        }

        if (lengthOf(avgFiles) > 0) {
            run("Images to Stack", "name=Stack title=[] use");
            originalStack = getImageID();
            columns = 4;
            rows = floor((nSlices + columns - 1) / columns);
            run("Make Montage...", "columns=" + columns + " rows=" + rows + " scale=1.0");
            run("Maximize");
        } else {
            print("[WARN] No average images found for montage creation.");
        }
        
        disableBatchMode();
        
        Dialog.create("結果確認");
        Dialog.addMessage("重ね合わせの結果は適切ですか?");
        Dialog.addChoice("選択:", newArray("はい", "いいえ、パラメータを調整する"), "はい");
        Dialog.show();
        userChoice = Dialog.getChoice();

        print(" User choice: " + userChoice);

        if (userChoice == "いいえ、パラメータを調整する") {
            print("[INFO] User requested parameter adjustment - preparing preview and dialog");
            run("Close All");

			if (lengthOf(listSubSubDirs) > 0) {
			    // 各サブサブディレクトリから最初の画像を開く
			    for (j = 0; j < lengthOf(listSubSubDirs); j++) {
			        subsubDirPath = subDirPath + listSubSubDirs[j];
			        images = getFileList(subsubDirPath);
			        
			        // 最初の画像ファイルを探す
			        for (k = 0; k < lengthOf(images); k++) {
			            if (endsWith(images[k], ".jpg") || endsWith(images[k], ".tif")) {
			                open(subsubDirPath + images[k]);
			                break;  // 最初の1枚だけ開いたらループを抜ける
			            }
			        }
			    }
			    
			    // 開いた画像が1枚以上あればモンタージュを作成
			    if (nImages > 0) {
			        run("Images to Stack", "name=file title=[] use");
			        originalStack = getImageID();
			        columns = 4;
			        rows = floor((nSlices + columns - 1) / columns);
			        run("Make Montage...", "columns=" + columns + " rows=" + rows + " scale=1.0");
			        
			        // スタックを閉じる
			        if (isOpen(originalStack)) {
			            selectImage(originalStack);
			            close();
			        }
			    } else {
			        print("[WARN] No preview images found");
			    }
			} else {
			    print("[WARN] listSubSubDirs undefined or empty - skipping preview");
			}
			
            Dialog.create("パラメータ調整");
            Dialog.addChoice("変換タイプ:", newArray("Translation", "Rigid Body", "Scaled Rotation", "Affine"), transformation);
            Dialog.addChoice("参照画像:", listStackTitles, reference);
            Dialog.addNumber("Exclude Folders?", skip);
            Dialog.show();
            transformation = Dialog.getChoice();
            reference = Dialog.getChoice();
            skip = Dialog.getNumber();

            print(" New parameters -> transformation: " + transformation + ", reference: " + reference + ", folder exclusion (number): " + skip);

            run("Close All");
        } else {
            print("[INFO] User confirmed results acceptable.");
        }

    } while (userChoice == "いいえ、パラメータを調整する");

    run("Close All");
    
    reregEndTime = getTime();
    reregElapsed = (reregEndTime - reregStartTime) / 1000;
    
    print("\n=================================================");
    print("=== Re-registration finished for visit: " + visitNumber + " ===");
    print("End timestamp: " + getDateTimeString());
    print("Re-registration time: " + d2s(reregElapsed, 2) + " seconds");
    print("=================================================");
}

///////////////////////////////////////////////////////////////////////////////
// ヘルパー関数
///////////////////////////////////////////////////////////////////////////////
function replace(s, search, repl) {
    var out = s;
    while (indexOf(out, search) >= 0) {
        out = substring(out, 0, indexOf(out, search)) + repl + substring(out, indexOf(out, search) + lengthOf(search));
    }
    return out;
}

function isOpen(title) {
    var l = getList("image.titles");
    for (var ii = 0; ii < lengthOf(l); ii++) {
        if (l[ii] == title) {
            return true;
        }
    }
    return false;
}

function arrayContains(arr, value) {
    for (var i = 0; i < lengthOf(arr); i++) {
        if (arr[i] == value) {
            return true;
        }
    }
    return false;
}

function pretreat(){
    // Step 1: Subtract background
    run("Subtract Background...", "rolling=50");

    // Step 2: Enhance brightness/contrast
    run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None* fast_(less_accurate)");

    // Step 3: Apply a Gaussian blur (optional, for noise reduction)
    run("Gaussian Blur...", "sigma=2");
}
// ===== バッチモード管理関数 =====

function enableBatchMode() {
    if (!isBatchModeActive) {
        setBatchMode(true);
        isBatchModeActive = true;
        print("[Batch Mode] Enabled");
    }
}

function disableBatchMode() {
    if (isBatchModeActive) {
        setBatchMode(false);
        isBatchModeActive = false;
        print("[Batch Mode] Disabled");
    }
}
