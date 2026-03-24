function mec13 = importfile(workbookFile, sheetName, dataLines)
%IMPORTFILE1 导入电子表格中的数据
%  MEC13 = IMPORTFILE1(FILE) 读取名为 FILE 的 Microsoft Excel
%  电子表格文件的第一张工作表中的数据。  返回数值数据。
%
%  MEC13 = IMPORTFILE1(FILE, SHEET) 从指定的工作表中读取。
%
%  MEC13 = IMPORTFILE1(FILE, SHEET,
%  DATALINES)按指定的行间隔读取指定工作表中的数据。对于不连续的行间隔，请将 DATALINES 指定为正整数标量或 N×2
%  正整数标量数组。
%
%  示例:
%  mec13 = importfile1("E:\152-school\OneDrive - bjtu.edu.cn\论文项目\周雨论文\python\MEC\picture\gpd\mec13.xlsx", "Sheet1", [1, 13]);
%
%  另请参阅 READTABLE。
%
% 由 MATLAB 于 2020-11-28 22:09:12 自动生成

%% 输入处理

% 如果未指定工作表，则将读取第一张工作表
if nargin == 1 || isempty(sheetName)
    sheetName = 1;
end

% 如果未指定行的起点和终点，则会定义默认值。
if nargin <= 2
    dataLines = [1, 13];
end

%% 设置导入选项并导入数据
opts = spreadsheetImportOptions("NumVariables", 3);

% 指定工作表和范围
opts.Sheet = sheetName;
opts.DataRange = "A" + dataLines(1, 1) + ":C" + dataLines(1, 2);

% 指定列名称和类型
opts.VariableNames = ["VarName1", "VarName2", "VarName3"];
opts.VariableTypes = ["double", "double", "double"];

% 导入数据
mec13 = readtable(workbookFile, opts, "UseExcel", false);

for idx = 2:size(dataLines, 1)
    opts.DataRange = "A" + dataLines(idx, 1) + ":C" + dataLines(idx, 2);
    tb = readtable(workbookFile, opts, "UseExcel", false);
    mec13 = [mec13; tb]; %#ok<AGROW>
end

%% 转换为输出类型
mec13 = table2array(mec13);
end