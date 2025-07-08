% --- GLRA-MOBLP 主控制脚本 ---
function glra_moblp_main()
    % 主要参数设置
    params.n_global_samples_s1 = 20;      % 阶段一：上层全局采样点数量
    params.n_ll_random_samples_s1 = 50; % 阶段一：下层随机采样点数 (用于近似Pareto)
    params.n_elite_candidates = 5;        % 阶段一：选择的候选点数量
    params.local_max_iters_s2 = 30;       % 阶段二：局部优化最大迭代次数
    params.learning_rate_s2 = 0.2;        % 阶段二：局部优化学习率
    params.n_ll_random_samples_s2 = 30; % 阶段二：下层随机采样点数
    params.total_glra_stages = 2;         % GLRA-MOBLP总迭代轮数

    % 初始化档案库
    refined_archive = []; % 存储最终高质量解 {x_ul, F_ul_value, risk_model, y_ll_objectives_chosen}
    global_archive_overall = []; % 存储所有探索到的点

    fprintf('====== GLRA-MOBLP 概念性求解器开始 ======\n');

    for stage_iter = 1:params.total_glra_stages
        fprintf('\n--- GLRA-MOBLP 总迭代阶段 %d/%d ---\n', stage_iter, params.total_glra_stages);

        % --- 阶段一：全局诱导集探索与候选区域识别 ---
        [candidate_x_ul, global_archive_iter] = stage1_global_exploration(params, global_archive_overall);
        
        if ~isempty(global_archive_iter)
            if isempty(global_archive_overall)
                global_archive_overall = global_archive_iter;
            else
                % 简单合并，实际中可能需要更智能的更新策略
                global_archive_overall = [global_archive_overall; global_archive_iter];
                % 可以考虑去重或保留精英，这里省略
            end
        end

        if isempty(candidate_x_ul)
            fprintf('  阶段一未能识别候选点。\n');
            if ~isempty(refined_archive)
                fprintf('  尝试从历史精炼档案中选择候选点。\n');
                % 按F_ul_value排序并选择最好的几个
                [~, sort_idx] = sort([refined_archive.F_ul_value]);
                temp_refined_sorted = refined_archive(sort_idx);
                num_to_select = min(params.n_elite_candidates, length(temp_refined_sorted));
                candidate_x_ul_struct = [temp_refined_sorted(1:num_to_select).x_ul];
                candidate_x_ul = cell2mat(candidate_x_ul_struct); % 确保是数值数组
                 if isempty(candidate_x_ul)
                    fprintf('  历史档案也为空或无法提取x_ul，算法终止。\n');
                    break;
                 end
            else
                fprintf('  全局和精炼档案均无候选，算法终止。\n');
                break;
            end
        end
        fprintf('  识别出的候选上层决策点 X_cand: %s\n', mat2str(candidate_x_ul,3));

        % --- 阶段二：基于风险偏好的局部梯度优化与解集增强 ---
        % 为简化，这里只演示 "optimistic" 风险模型
        risk_models_to_try = ["optimistic"]; % 可以扩展到 "risk_neutral", "risk_averse"

        for i = 1:length(candidate_x_ul)
            x_c = candidate_x_ul(i);
            for r_idx = 1:length(risk_models_to_try)
                current_risk_model = risk_models_to_try(r_idx);
                fprintf('\n  开始对候选点 x_ul=%.2f 进行局部优化 (风险模型: %s)\n', x_c, current_risk_model);
                
                local_opt_result = stage2_local_optimization(x_c, current_risk_model, params);
                
                if ~isempty(local_opt_result)
                    refined_archive = [refined_archive; local_opt_result];
                end
            end
        end
        
        % 更新和筛选 refined_archive (保留最好的几个解)
        if ~isempty(refined_archive)
            [~, sort_idx] = sort([refined_archive.F_ul_value]);
            refined_archive = refined_archive(sort_idx);
            num_to_keep = min(10, length(refined_archive)); % 保留前10个
            refined_archive = refined_archive(1:num_to_keep);
            
            fprintf('\n  当前精炼档案库中的最佳解 (部分):\n');
            for k_idx = 1:min(3, length(refined_archive))
                item = refined_archive(k_idx);
                fprintf('    x_ul=%.4f, F_ul=%.4f, model=%s, y_ll_chosen_objs=[%.2f, %.2f]\n', ...
                        item.x_ul, item.F_ul_value, item.risk_model, item.y_ll_objectives_chosen(1), item.y_ll_objectives_chosen(2));
            end
        else
            fprintf('  精炼档案库为空。\n');
        end
    end

    fprintf('\n====== GLRA-MOBLP 概念性求解器结束 ======\n');
    if ~isempty(refined_archive)
        fprintf('最终精炼档案库中的最佳解:\n');
        best_overall_solution = refined_archive(1); % 因为已经排序
        fprintf('  x_ul=%.4f, F_ul=%.4f, model=%s, y_ll_chosen_objs=[%.2f, %.2f]\n', ...
                best_overall_solution.x_ul, best_overall_solution.F_ul_value, ...
                best_overall_solution.risk_model, best_overall_solution.y_ll_objectives_chosen(1), best_overall_solution.y_ll_objectives_chosen(2));
        
        % 简单的绘图
        figure;
        hold on;
        if ~isempty(global_archive_overall)
            plot([global_archive_overall.x_ul], [global_archive_overall.F_ul_value], 'b.', 'MarkerSize', 10, 'DisplayName', '全局探索点');
        end
        plot([refined_archive.x_ul], [refined_archive.F_ul_value], 'ro', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', '精炼解');
        if ~isempty(best_overall_solution)
             plot(best_overall_solution.x_ul, best_overall_solution.F_ul_value, 'k*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', '最佳解');
        end
        xlabel('上层决策 x_{ul}');
        ylabel('上层目标函数 F_{ul}');
        title('GLRA-MOBLP 求解过程示意');
        legend show;
        grid on;
        hold off;
        
    else
        fprintf('算法未能找到任何解。\n');
    end
end

% --- 阶段一函数 ---
function [candidate_x_ul_values, global_archive] = stage1_global_exploration(params, existing_global_archive)
    fprintf('\n--- GLRA-MOBLP: 阶段一 开始 ---\n');
    global_archive = []; % 本轮迭代的全局档案

    % 1. 上层决策空间采样
    x_ul_samples = linspace(0, 10, params.n_global_samples_s1);
    fprintf('  上层采样点数量: %d\n', length(x_ul_samples));

    for i = 1:length(x_ul_samples)
        x_s = x_ul_samples(i);
        
        % 2. 下层Pareto前沿近似 (通过随机采样和非支配排序)
        [y_ll_pareto_vars, y_ll_pareto_objectives] = solve_lower_level_approx(x_s, params.n_ll_random_samples_s1);
        
        if ~isempty(y_ll_pareto_vars)
            for j = 1:size(y_ll_pareto_vars, 1) % 对每个找到的下层Pareto近似解
                y_ll_vars_sol = y_ll_pareto_vars(j, :);
                y_ll_objectives_sol = y_ll_pareto_objectives(j, :);
                
                f_ul_value = upper_level_objective_func(x_s, y_ll_objectives_sol);
                
                current_entry.x_ul = x_s;
                current_entry.y_ll_vars = y_ll_vars_sol;
                current_entry.y_ll_objectives = y_ll_objectives_sol;
                current_entry.F_ul_value = f_ul_value;
                global_archive = [global_archive; current_entry];
            end
        end
    end

    if isempty(global_archive)
        fprintf('  全局探索阶段未能生成任何有效解。\n');
        candidate_x_ul_values = [];
        return;
    end

    % 4. 候选区域/点识别 (简化：选择F_ul_value最小的几个x_ul)
    [~, sort_idx] = sort([global_archive.F_ul_value]);
    sorted_archive = global_archive(sort_idx);
    
    candidate_x_ul_values_cell = {};
    seen_x_ul = [];
    count = 0;
    for k = 1:length(sorted_archive)
        if ~ismember(sorted_archive(k).x_ul, seen_x_ul)
            candidate_x_ul_values_cell{end+1} = sorted_archive(k).x_ul;
            seen_x_ul = [seen_x_ul, sorted_archive(k).x_ul];
            count = count + 1;
            if count >= params.n_elite_candidates
                break;
            end
        end
    end
    candidate_x_ul_values = cell2mat(candidate_x_ul_values_cell);


    fprintf('  全局探索完成。本轮档案库大小: %d。\n', length(global_archive));
    fprintf('--- GLRA-MOBLP: 阶段一 结束 ---\n');
end

% --- 阶段二函数 ---
function local_opt_result = stage2_local_optimization(x_candidate, risk_model, params)
    x_current = x_candidate;
    best_x_local = x_current;
    
    % 初始评估
    [~, y_ll_obj_init_chosen] = solve_ll_and_select_for_ul(x_current, risk_model, params.n_ll_random_samples_s2);
    if isempty(y_ll_obj_init_chosen)
        best_f_ul_local = inf; % 如果下层无解
    else
        best_f_ul_local = upper_level_objective_func(x_current, y_ll_obj_init_chosen);
    end
    
    fprintf('    初始评估: x_ul=%.2f, F_ul_value=%.4f (风险模型: %s)\n', x_current, best_f_ul_local, risk_model);

    history_x = x_current;
    history_F = best_f_ul_local;

    for iter_num = 1:params.local_max_iters_s2
        % 1. 定义局部目标函数 phi_local(x) - 高度依赖风险模型
        % 2. 计算其梯度 nabla_phi_local(x_current)
        %    *** 这是最复杂的部分，实际实现需要Giovanelli论文中的梯度公式 ***
        %    *** 这里用一个非常简化的模拟梯度，不代表真实计算 ***
        mock_gradient = 0.2 * (x_current - 5.0) + randn()*0.01; % 引导x_current向5靠近，加小噪声
        
        % 针对不同风险模型的模拟调整（非常概念性）
        if strcmp(risk_model, "risk_neutral")
            mock_gradient = mock_gradient + randn()*0.05; % 增加随机性
        elseif strcmp(risk_model, "risk_averse")
            % 风险规避可能更保守，减小步长或梯度
            mock_gradient = mock_gradient * 0.8 - sign(x_current - 5.0) * 0.02;
        end
            
        % 3. 变量更新
        x_next = x_current - params.learning_rate_s2 * mock_gradient;
        
        % 4. 投影回可行域 (假设 x_ul 在 [0, 10])
        x_next = max(0, min(10, x_next));
        
        % 5. 评估新的 x_next (包括求解下层问题)
        [~, y_ll_obj_next_chosen] = solve_ll_and_select_for_ul(x_next, risk_model, params.n_ll_random_samples_s2);
        if isempty(y_ll_obj_next_chosen)
            f_ul_value_next = inf;
        else
            f_ul_value_next = upper_level_objective_func(x_next, y_ll_obj_next_chosen);
        end
        
        % fprintf('      迭代 %d: x_ul=%.4f, F_ul=%.4f (模拟梯度: %.4f)\n', iter_num, x_next, f_ul_value_next, mock_gradient);

        if f_ul_value_next < best_f_ul_local
            best_f_ul_local = f_ul_value_next;
            best_x_local = x_next;
            best_y_ll_obj_chosen_local = y_ll_obj_next_chosen;
        else
             % 如果没有改善，可能考虑减小学习率或接受概率（模拟退火思想），这里简化
        end
        
        history_x = [history_x, x_next];
        history_F = [history_F, f_ul_value_next];
        
        if abs(x_next - x_current) < 1e-3 && iter_num > 5 % 至少迭代几次
            fprintf('    局部优化收敛于迭代 %d.\n', iter_num);
            break;
        end
        x_current = x_next;
    end
    
    if isinf(best_f_ul_local) || isempty(best_y_ll_obj_chosen_local) % 确保有有效的下层解
        local_opt_result = [];
        fprintf('    局部优化未能找到有效解 (下层可能无解或评估失败)。\n');
    else
        local_opt_result.x_ul = best_x_local;
        local_opt_result.F_ul_value = best_f_ul_local;
        local_opt_result.risk_model = risk_model;
        local_opt_result.y_ll_objectives_chosen = best_y_ll_obj_chosen_local; % 存储选择的下层目标值
        fprintf('    局部优化完成。最佳x_ul=%.4f, 对应上层目标值=%.4f\n', best_x_local, best_f_ul_local);
    end
end

% --- 下层问题求解器 (非常简化的近似) ---
function [pareto_vars, pareto_objectives] = solve_lower_level_approx(x_ul, n_samples)
    % 通过随机采样和非支配排序来近似Pareto前沿
    % 下层变量 y_ll = [y1, y2], 范围 [0,1]
    n_ll_vars = 2;
    sampled_y_ll = rand(n_samples, n_ll_vars); % 在[0,1]之间随机采样
    
    objectives = zeros(n_samples, 2);
    for i = 1:n_samples
        objectives(i, :) = lower_level_objectives_func(sampled_y_ll(i, :), x_ul);
    end
    
    % 非支配排序 (简化版：只找第一层非支配解)
    is_dominated_flag = false(n_samples, 1);
    for i = 1:n_samples
        for j = 1:n_samples
            if i == j
                continue;
            end
            % 如果 objectives(i,:) 被 objectives(j,:) 支配
            if all(objectives(j, :) <= objectives(i, :)) && any(objectives(j, :) < objectives(i, :))
                is_dominated_flag(i) = true;
                break;
            end
        end
    end
    
    pareto_indices = ~is_dominated_flag;
    pareto_vars = sampled_y_ll(pareto_indices, :);
    pareto_objectives = objectives(pareto_indices, :);
    
    % 如果没有非支配解（不太可能，除非所有解都相同），返回所有采样点
    if isempty(pareto_vars) && n_samples > 0
        pareto_vars = sampled_y_ll;
        pareto_objectives = objectives;
    end
end

% --- 根据风险模型从下层Pareto近似中选择一个解的目标值 ---
function [y_ll_vars_chosen, y_ll_objectives_chosen] = solve_ll_and_select_for_ul(x_ul, risk_model, n_ll_samples)
    [ll_pareto_vars, ll_pareto_objectives] = solve_lower_level_approx(x_ul, n_ll_samples);
    
    y_ll_vars_chosen = [];
    y_ll_objectives_chosen = [];

    if isempty(ll_pareto_vars)
        return; % 如果下层没有找到近似Pareto解
    end
    
    if strcmp(risk_model, "optimistic")
        % 乐观：选择使上层目标F_ul最小的那个下层Pareto解
        f_ul_values_for_pareto_sols = zeros(size(ll_pareto_objectives, 1), 1);
        for i = 1:size(ll_pareto_objectives, 1)
            f_ul_values_for_pareto_sols(i) = upper_level_objective_func(x_ul, ll_pareto_objectives(i, :));
        end
        [~, min_idx] = min(f_ul_values_for_pareto_sols);
        y_ll_vars_chosen = ll_pareto_vars(min_idx, :);
        y_ll_objectives_chosen = ll_pareto_objectives(min_idx, :);
        
    elseif strcmp(risk_model, "risk_neutral")
        % 风险中性 (极度简化)：随机选一个，或者取平均？这里随机选一个
        rand_idx = randi(size(ll_pareto_objectives, 1));
        y_ll_vars_chosen = ll_pareto_vars(rand_idx, :);
        y_ll_objectives_chosen = ll_pareto_objectives(rand_idx, :);
        
    elseif strcmp(risk_model, "risk_averse")
        % 风险规避 (极度简化)：选择使上层目标F_ul最大的那个下层Pareto解
        f_ul_values_for_pareto_sols = zeros(size(ll_pareto_objectives, 1), 1);
        for i = 1:size(ll_pareto_objectives, 1)
            f_ul_values_for_pareto_sols(i) = upper_level_objective_func(x_ul, ll_pareto_objectives(i, :));
        end
        [~, max_idx] = max(f_ul_values_for_pareto_sols);
        y_ll_vars_chosen = ll_pareto_vars(max_idx, :);
        y_ll_objectives_chosen = ll_pareto_objectives(max_idx, :);
    else
        % 默认乐观
        f_ul_values_for_pareto_sols = zeros(size(ll_pareto_objectives, 1), 1);
        for i = 1:size(ll_pareto_objectives, 1)
            f_ul_values_for_pareto_sols(i) = upper_level_objective_func(x_ul, ll_pareto_objectives(i, :));
        end
        [~, min_idx] = min(f_ul_values_for_pareto_sols);
        y_ll_vars_chosen = ll_pareto_vars(min_idx, :);
        y_ll_objectives_chosen = ll_pareto_objectives(min_idx, :);
    end
end


% --- 示例问题定义：上层目标函数 ---
function F_ul = upper_level_objective_func(x_ul, y_ll_objectives)
    % x_ul: 上层决策变量 (标量)
    % y_ll_objectives: 下层问题的目标函数值 [f1_ll, f2_ll] (1x2 向量)
    if isempty(y_ll_objectives) || any(isinf(y_ll_objectives)) || any(isnan(y_ll_objectives))
        F_ul = inf; % 如果下层解无效，则上层目标为无穷大
        return;
    end
    F_ul = 0.1 * (x_ul - 5)^2 + y_ll_objectives(1) + 0.5 * y_ll_objectives(2);
end

% --- 示例问题定义：下层目标函数 ---
function f_ll = lower_level_objectives_func(y_ll, x_ul)
    % y_ll: 下层决策变量 [y1, y2] (1x2 向量)
    % x_ul: 上层决策变量 (标量)
    y1 = y_ll(1);
    y2 = y_ll(2); % 在这个简化示例中，y2用于计算g，但g的公式简化了
    
    f1 = y1 * (1 + 0.1 * x_ul);
    
    % 简化的g函数，只依赖y2 (原ZDT1依赖y_ll(2)...y_ll(end))
    % 为确保g不为0，且sqrt内部非负
    % 假设y2在[0,1], g_simplified = 1 + 9*y2;
    g_simplified = 1 + 9 * y2; % 确保 g > 0
    
    if g_simplified < 1e-6 % 避免除以零
        f2 = inf;
    else
        ratio = f1 / g_simplified;
        if ratio < 0 % 理论上不会，因为y1, x_ul, g_simplified都非负
            ratio = 0;
        end
        f2 = g_simplified * (1 - sqrt(ratio));
    end
    f_ll = [f1, f2];
end
