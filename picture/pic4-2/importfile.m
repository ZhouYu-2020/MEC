function [UE] = importfile(workbookFile, sheetName, dataLines)
%IMPORTFILE 导入电子表格中的数据
%  [UE] = IMPORTFILE(FILE) 读取名为 FILE 的 Microsoft Excel
%  电子表格文件的第一张工作表中的数据。  以列向量形式返回数据。
%
%  [UE] = IMPORTFILE(FILE, SHEET) 从指定的工作表中读取。
%
%  [UE] = IMPORTFILE(FILE, SHEET,
%  DATALINES)按指定的行间隔读取指定工作表中的数据。对于不连续的行间隔，请将 DATALINES 指定为正整数标量或 N×2
%  正整数标量数组。
%
%  示例:
%  [UE] = importfile("D:\OneDrive - bjtu.edu.cn\论文项目\周雨论文\python\MEC\picture\pic1\mec.xlsx", "Sheet1", [2, 2001]);
%
%  另请参阅 READTABLE。
%
% 由 MATLAB 于 2020-11-24 17:51:43 自动生成

%% 输入处理

% 如果未指定工作表，则将读取第一张工作表
if nargin == 1 || isempty(sheetName)
    sheetName = 1;
end

% 如果未指定行的起点和终点，则会定义默认值。
if nargin <= 2
    dataLines = [2, 2001];
end

%% 设置导入选项并导入数据
opts = spreadsheetImportOptions("NumVariables", 1);

% 指定工作表和范围
opts.Sheet = sheetName;
opts.DataRange = "A" + dataLines(1, 1) + ":A" + dataLines(1, 2);

% 指定列名称和类型
opts.VariableNames = "UE";
opts.VariableTypes = "double";

% 导入数据
tbl = readtable(workbookFile, opts, "UseExcel", false);

for idx = 2:size(dataLines, 1)
    opts.DataRange = "A" + dataLines(idx, 1) + ":A" + dataLines(idx, 2);
    tb = readtable(workbookFile, opts, "UseExcel", false);
    tbl = [tbl; tb]; %#ok<AGROW>
end

%% 转换为输出类型
UE = tbl.UE;
end