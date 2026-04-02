% 文件路径（请根据实际情况修改）
csvFile = 'results_paper_c5/stress_scan_summary.csv';
outDir = 'figures_paper_c5';

% ==================== 检测分隔符 ====================
fid = fopen(csvFile, 'r', 'n', 'UTF-8');
if fid == -1
    error('无法打开文件: %s', csvFile);
end
firstLine = fgetl(fid);
fclose(fid);

% 去除 BOM
if length(firstLine) >= 1 && double(firstLine(1)) == 65279
    firstLine = firstLine(2:end);
end

% 自动检测分隔符
delimiters = {',', '\t', ' ', ';', '|'};
delim = '';
maxCols = 1;
for i = 1:length(delimiters)
    parts = strsplit(firstLine, delimiters{i});
    if length(parts) > maxCols
        maxCols = length(parts);
        delim = delimiters{i};
    end
end
if isempty(delim)
    error('未检测到分隔符');
end
fprintf('使用分隔符: "%s"\n', delim);

% ==================== 读取全部数据 ====================
fid = fopen(csvFile, 'r', 'n', 'UTF-8');
if fid == -1
    error('无法打开文件: %s', csvFile);
end
headerLine = fgetl(fid);
if double(headerLine(1)) == 65279
    headerLine = headerLine(2:end);
end
headers = strsplit(headerLine, delim);
headers = strtrim(headers);
nCols = length(headers);

dataLines = {};
rowCount = 0;
while ~feof(fid)
    line = fgetl(fid);
    rowCount = rowCount + 1;
    if isempty(strtrim(line))
        continue;
    end
    parts = strsplit(line, delim);
    if length(parts) ~= nCols
        warning('行 %d 列数 (%d) 与表头 (%d) 不一致，跳过', rowCount+1, length(parts), nCols);
        continue;
    end
    dataLines = [dataLines; parts]; %#ok<AGROW>
end
fclose(fid);

% 转换为表格
dataTable = table();
for c = 1:nCols
    colName = headers{c};
    colData = dataLines(:, c);
    numData = str2double(colData);
    if all(~isnan(numData))
        dataTable.(colName) = numData;
    else
        dataTable.(colName) = colData;
    end
end

% 所需列名（小写）
requiredCols = {'scenario', 'value', 'policy', 'p_c6_viol_mean', ...
                'p_c7_viol_mean', 'p_c8_viol_mean'};

% 列名映射（忽略大小写）
actualCols = dataTable.Properties.VariableNames;
colMap = containers.Map();
for i = 1:length(actualCols)
    key = lower(strtrim(actualCols{i}));
    colMap(key) = actualCols{i};
end

% 提取所需列
missing = {};
selectedCols = {};
for i = 1:length(requiredCols)
    req = requiredCols{i};
    if isKey(colMap, req)
        selectedCols{end+1} = colMap(req);
    else
        missing{end+1} = req;
    end
end
if ~isempty(missing)
    error('缺少列: %s\n实际列名: %s', strjoin(missing, ', '), strjoin(actualCols, ', '));
end
T_sel = dataTable(:, selectedCols);
T_sel.Properties.VariableNames = requiredCols;

% 筛选 scenario = 'user_num'
if isnumeric(T_sel.scenario)
    T_sel.scenario = string(T_sel.scenario);
end
T_sel = T_sel(strcmp(T_sel.scenario, 'user_num'), :);
if isempty(T_sel)
    error('未找到 scenario = user_num 的数据');
end
fprintf('筛选后剩余 %d 行\n', height(T_sel));

% 按 (value, policy) 聚合
[G, valueGroup, policyGroup] = findgroups(T_sel.value, T_sel.policy);
c6_mean = splitapply(@mean, T_sel.p_c6_viol_mean, G);
c7_mean = splitapply(@mean, T_sel.p_c7_viol_mean, G);
c8_mean = splitapply(@mean, T_sel.p_c8_viol_mean, G);
agg = table(valueGroup, policyGroup, c6_mean, c7_mean, c8_mean, ...
    'VariableNames', {'value', 'policy', 'c6', 'c7', 'c8'});

% ==================== 仅保留指定策略 ====================
keepPolicies = {'Full', 'No-EVT', 'Non-MEC', 'Stand-alone MEC', 'Random'};
agg = agg(ismember(agg.policy, keepPolicies), :);
if isempty(agg)
    error('没有找到指定的策略数据。请检查策略名称是否与CSV中的一致。');
end
% 保持策略顺序为 keepPolicies 中存在的顺序
policies = keepPolicies(ismember(keepPolicies, unique(agg.policy)));
fprintf('保留的策略: %s\n', strjoin(policies, ', '));

% 重新获取 values（可能因过滤而减少，但通常不变）
values = sort(unique(agg.value));

% ==================== 绘图 ====================
fig = figure('Position', [100 100 1500 460]);
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
metrics = {'c6', 'c7', 'c8'};
titles = {'C6违约率', 'C7违约率', 'C8违约率'};
colors = lines(numel(policies));
markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', 'x', '+'};

for i = 1:3
    nexttile;
    hold on;
    for pIdx = 1:numel(policies)
        policy = policies{pIdx};
        subData = agg(strcmp(agg.policy, policy), :);
        [~, idx] = ismember(values, subData.value);
        y = nan(size(values));
        y(idx) = subData.(metrics{i});
        plot(values, y, 'LineWidth', 1.8, ...
            'Marker', markers{mod(pIdx-1, numel(markers))+1}, ...
            'MarkerSize', 5.5, 'Color', colors(pIdx,:), ...
            'DisplayName', policy);
    end
    hold off;
%     title(titles{i}, 'FontWeight', 'bold');
    xlabel('设备数');
    ylabel(titles{i});
    ylim([0 1]);
    grid on;
end


lgd = legend('Location', 'best', 'NumColumns', 1);
lgd.Box = 'on';


% 保存图片（兼容旧版 MATLAB）
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
outFile = fullfile(outDir, 'c6_c8_vs_user_num_line.png');
if exist('exportgraphics', 'file')
    exportgraphics(fig, outFile, 'Resolution', 300);
else
    print(fig, outFile, '-dpng', '-r300');
end
fprintf('图片已保存至: %s\n', outFile);
