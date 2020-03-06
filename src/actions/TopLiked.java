package actions;

import actions.util.ListNode;

import java.util.Stack;

public class TopLiked {
    public boolean isMatch(String s, String p) {
        if (p == null || p.length() == 0) {
            return s == null || s.length() == 0;
        }
        if (s == null || s.length() == 0) {
            return p.length() == 0;
        }
        int[][] dp = new int[s.length()][p.length()];
        return isMatchTopDown(s, 0, p, 0, dp);
    }

    private boolean isMatchTopDown(String s, int i, String p, int j, int[][] dp) {
        if (i == s.length()) {
            while (j < p.length()) {
                if (p.charAt(j) != '*') {
                    return false;
                }
            }
            return true;
        }
        if (j == p.length()) {
            return false;
        }
        if (dp[i][j] != 0) {
            return dp[i][j] == 1;
        }

        boolean result = false;
        if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.') {
            result = isMatchTopDown(s, i + 1, p, j + 1, dp);
        } else if (p.charAt(j) == '*') {
            result = isMatchTopDown(s, i, p, j + 1, dp) || isMatchTopDown(s, i + 1, p, j, dp);

        }

        if (result) {
            dp[i][j] = 1;
        } else {
            dp[i][j] = -1;
        }

        return result;
    }

    public boolean isMatch2(String s, String p) {
        int[][] dp = new int[s.length()][p.length()];
        return isMatch2TopDown(s, 0, p, 0, dp);
    }

    private boolean isMatch2TopDown(String s, int i, String p, int j, int[][] dp) {
        if (i == s.length()) {
            while (j < p.length()) {
                if (j == p.length() - 1 || p.charAt(j + 1) != '*') {
                    return false;
                }
                j += 2;
            }
            return true;
        }
        if (j == p.length()) {
            return false;
        }
        if (dp[i][j] != 0) {
            return dp[i][j] == 1;
        }

        boolean result = false;
        if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.') {
            if (j < p.length() - 1 && p.charAt(j + 1) == '*') {
                result = isMatch2TopDown(s, i + 1, p, j, dp) || isMatch2TopDown(s, i, p, j + 2, dp);
            } else {
                result = isMatch2TopDown(s, i + 1, p, j + 1, dp);
            }
        } else if (j < p.length() - 1 && p.charAt(j + 1) == '*') {
            result = isMatch2TopDown(s, i, p, j + 2, dp);
        }

        if (result) {
            dp[i][j] = 1;
        } else {
            dp[i][j] = -1;
        }
        return result;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode first = dummy;
        ListNode second = dummy;
        while (n >= 0) {
            first = first.next;
            n--;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return dummy.next;
    }

    public int longestValidParentheses(String s) {
        int maxans = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.empty()) {
                    stack.push(i);
                } else {
                    maxans = Math.max(maxans, i - stack.peek());
                }
            }
        }
        return maxans;
    }
}
