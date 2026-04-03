% 文件路径（请根据实际情况修改）
csvFile = 'results_2000ep_3seed_20260403/stress_scan_summary.csv';
outDir = 'figures_2000ep_3seed_20260403';

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

% ==================== 生成图A与图C ====================
plot_c6_c8(T_sel, 'user_num', '设备数', 'c6_c8_vs_user_num_line.png', outDir);
plot_c6_c8(T_sel, 'f_scale', '计算资源缩放系数', 'c6_c8_vs_f_scale_line.png', outDir);

% ==================== 通用绘图函数 ====================
function plot_c6_c8(T_sel, scenarioName, xLabel, outFile, outDir)
    % 过滤指定场景
    if isnumeric(T_sel.scenario)
        T_sel.scenario = string(T_sel.scenario);
    end
    T_s = T_sel(strcmp(T_sel.scenario, scenarioName), :);
    if isempty(T_s)
        error('未找到 scenario = %s 的数据', scenarioName);
    end

    % 按 (value, policy) 聚合
    [G, valueGroup, policyGroup] = findgroups(T_s.value, T_s.policy);
    c6_mean = splitapply(@mean, T_s.p_c6_viol_mean, G);
    c7_mean = splitapply(@mean, T_s.p_c7_viol_mean, G);
    c8_mean = splitapply(@mean, T_s.p_c8_viol_mean, G);
    agg = table(valueGroup, policyGroup, c6_mean, c7_mean, c8_mean, ...
        'VariableNames', {'value', 'policy', 'c6', 'c7', 'c8'});

    % 策略顺序
    keepPolicies = {'Full', 'IQL', 'No-EVT', 'C6-Only', 'Non-MEC', 'Stand-alone MEC', 'Random', 'Lyapunov-Greedy'};
    agg = agg(ismember(agg.policy, keepPolicies), :);
    if isempty(agg)
        error('没有找到指定的策略数据。请检查策略名称是否与CSV中的一致。');
    end
    policies = keepPolicies(ismember(keepPolicies, unique(agg.policy)));

    values = sort(unique(agg.value));

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
        xlabel(xLabel);
        ylabel(titles{i});
        ylim([0 1]);
        grid on;
    end

    lgd = legend('Location', 'best', 'NumColumns', 1);
    lgd.Box = 'on';

    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    outFileFull = fullfile(outDir, outFile);
    if exist('exportgraphics', 'file')
        exportgraphics(fig, outFileFull, 'Resolution', 300);
    else
        print(fig, outFileFull, '-dpng', '-r300');
    end
    fprintf('图片已保存至: %s\n', outFileFull);
end


