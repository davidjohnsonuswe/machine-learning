function [values, policy, nbr_pol_iter, nbr_pol_eval] = policy_iteration(pol_eval_tol, next_state_idxs, rewards, gamm)
%
% Code may be changed in this function, but only where it states that it is 
% allowed to do so. You need to implement policy iteration in this
% function.
%
% Function to run policy iteration to learn an optimal policy. Note that
% this implementation assumes eating apple is also a terminal state (and 
% not only hitting a wall / the body of the snake). Think about why this 
% is OK, i.e., why it will give rise to an optimal policy also for the 
% Snake game in which eating an apple is non-terminal (see also exercise 4).
%
% Input:
%
% pol_eval_tol    - Policy evaluation stopping tolerance.
% next_state_idxs - nbr_states-by-nbr_actions matrix; each entry of this 
%                   matrix is an integer in {-1, 0, 1, 2, ..., nbr_states}.
%                   In particular, the ith row of next_state_idxs gives
%                   the state indexes of taking the left, forward and right 
%                   actions. The only exceptions to this is if an action leads 
%                   to any terminal state; if an action leads to death, then
%                   the corresponding entry in next_state_idxs is 0; if an
%                   action leads to eating an apple, then the corresponding
%                   entry in next_state_idxs is -1.
% rewards         - Struct of the form struct('default', x, 'apple', y, 'death', z)
%                   Here x refers to the default reward, which is received
%                   when the snake only moves without dying and without
%                   eating an apple; y refers to the reward obtained when
%                   eating an apple; and z refers to the reward obtained when
%                   dying.
% gamm            - Discount factor in [0,1].
%
% Output:
%
% values       - 1-by-nbr_states vector; will after successful policy
%                iteration contain optimal values for all the non-terminal 
%                states.
% policy       - 1-by-nbr_states vector; will after successful policy
%                iteration contain optimal actions to take for all the 
%                non-terminal states.
% nbr_pol_iter - The number of iterations that the policy iteration runs
%                for; may be used e.g. for diagnostic purposes.
%                   
% Bugs, ideas etcetera: send them to the course email.

% Get number of non-terminal states and actions.
[nbr_states, nbr_actions] = size(next_state_idxs);

% Arbitrary initialization of values and policy.
values = randn(1, nbr_states);  
policy = randi(3, 1, nbr_states); % policy is size 1-by-nbr_states
                                  % the entries of policy are 1, 2 or 3
                                  % selected uniformly at random

% Counters over number of policy iterations and policy evaluations,
% for possible diagnostic purposes.
nbr_pol_iter = 0;
nbr_pol_eval = 0;

% This while-loop runs the policy iteration.
while 1
    
    % Policy evaluation.
    while 1
        
        Delta = 0;
        for state_idx = 1 : nbr_states
            value_old = values(state_idx);
        
            % Get the action chosen by the current policy.
            action = policy(state_idx);
            % Get the index of the next state.
            next_state_idx = next_state_idxs(state_idx,action);
            
            if next_state_idx == -1
                % Eat an apple.
                next_value = rewards.apple;
            elseif next_state_idx == 0
                % Death.
                next_value = rewards.death;
            else
                % Non-terminal state.
                next_value = rewards.default+gamm*values(next_state_idx);
            end
            
            % Update the value of the current state.
            values(state_idx) = next_value;
            
            % Update Delta
            Delta = max(Delta, abs(value_old-next_value));
        end
        
        % Increase nbr_pol_eval counter.
        nbr_pol_eval = nbr_pol_eval + 1;
        
        % Check for policy evaluation termination.
        if Delta < pol_eval_tol
            break;
        else
            disp(['Delta: ', num2str(Delta)])
        end
    end
    
    % Policy improvement.
    policy_stable = true;
    for state_idx = 1 : nbr_states
        old_action = policy(state_idx);
    
        % Evaluate the expected return for each action.
        action_returns = zeros(1, nbr_actions);
        for action = 1 : nbr_actions
            next_state_idx = next_state_idxs(state_idx,action);
            
            if next_state_idx == -1
                % Eat an apple.
                action_returns(action) = rewards.apple;
            elseif next_state_idx == 0
                % Death.
                action_returns(action) = rewards.death;
            else
                % Non-terminal state.
                action_returns(action) = rewards.default+gamm*values(next_state_idx);
            end
        end
        
        % Find the optimal action.
        [~, optimal_action] = max(action_returns);
        policy(state_idx) = optimal_action;
        
        % Check if the policy has changed.
        if old_action ~= optimal_action
            policy_stable = false;
        end
    end
    
    % Increase the number of policy iterations .
    nbr_pol_iter = nbr_pol_iter + 1;
    
    % Check for policy iteration termination (terminate if and only if the
    % policy is no longer changing, i.e. if and only if the policy is
    % stable).
    if policy_stable
        break;
    end
end
end