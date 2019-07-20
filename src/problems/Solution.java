package problems;

import java.util.*;

public class Solution {

    public int[] twoSum(int[] nums, int target) {
        int[] ans = new int[2];
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                ans[0] = map.get(complement);
                ans[1] = i;
                break;
            }
            map.put(nums[i], i);
        }

        return ans;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummyNode = new ListNode(0);
        ListNode previousNode = dummyNode;
        int carry = 0;

        while (l1 != null || l2 != null) {
            int val1 = (l1 != null) ? l1.val : 0;
            int val2 = l2 != null ? l2.val : 0;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }

            int temp_sum = val1 + val2 + carry;
            int val = temp_sum % 10;
            carry = temp_sum / 10;

            ListNode newNode = new ListNode(val);
            previousNode.next = newNode;
            previousNode = newNode;
        }

        if (carry != 0) {
            ListNode newNode = new ListNode(carry);
            previousNode.next = newNode;
        }

        return dummyNode.next;
    }

    public int lengthOfLongestSubstring(String s) {
        int result = 0;
        char[] chars = s.toCharArray();
        int left = 0, right = 0;
        HashSet<Character> set = new HashSet<Character>();

        while (right < chars.length) {
            if (set.contains(chars[right])) {
                set.remove(chars[left]);
                left++;
            } else {
                set.add(chars[right]);
                right++;
                if (right - left > result) {
                    result = right - left;
                }
            }
        }
        return result;
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if ((m + n) % 2 == 0) {
            return (double) (getkth(nums1, nums2, (m + n) / 2 + 1) + getkth(nums1, nums2, (m + n) / 2))
                    * 0.5;
        } else {
            return (double) (getkth(nums1, nums2, (m + n) / 2 + 1)) * 1.0;
        }
    }

    public int getkth(int[] A, int[] B, int k) {
        int m = A.length;
        int n = B.length;

        if (m > n) {
            return getkth(B, A, k);
        }
        if (m == 0) {
            return B[k - 1];
        }
        if (k == 1) {
            return Math.min(A[0], B[0]);
        }

        int pa = Math.min(k / 2, m);
        int pb = k - pa;
        if (A[pa - 1] <= B[pb - 1]) {
            return getkth(Arrays.copyOfRange(A, pa, m), B, pb);
        } else {
            return getkth(A, Arrays.copyOfRange(B, pb, n), pa);
        }
    }

    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }

        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length()) {
            if (s.charAt(left) != s.charAt(right)) {
                break;
            }
            left--;
            right++;
        }
        return right - left - 1;
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        backtrackSubsets(list, new ArrayList<Integer>(), nums, 0);
        return list;
    }

    private void backtrackSubsets(
            List<List<Integer>> list, List<Integer> tmplist, int[] nums, int start) {
        list.add(new ArrayList<Integer>(tmplist));
        for (int i = start; i < nums.length; i++) {
            tmplist.add(nums[i]);
            backtrackSubsets(list, tmplist, nums, i + 1);
            tmplist.remove(tmplist.size() - 1);
        }
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        backtrackSubsetsWithDup(result, new ArrayList<Integer>(), nums, 0);
        return result;
    }

    private void backtrackSubsetsWithDup(
            List<List<Integer>> result, List<Integer> tmplist, int[] nums, int start) {
        result.add(new ArrayList<>(tmplist));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            tmplist.add(nums[i]);
            backtrackSubsetsWithDup(result, tmplist, nums, i + 1);
            tmplist.remove(tmplist.size() - 1);
        }
    }

    public String convert(String s, int numRows) {
        if (numRows == 1) {
            return s;
        }

        List<StringBuilder> rows = new ArrayList<>();
        for (int i = 0; i < Math.min(s.length(), numRows); i++) {
            rows.add(new StringBuilder());
        }

        int curRow = 0;
        boolean goingDown = false;

        for (char c : s.toCharArray()) {
            rows.get(curRow).append(c);
            if (curRow == 0 || curRow == numRows - 1) {
                goingDown = !goingDown;
            }
            curRow += goingDown ? 1 : -1;
        }

        StringBuilder result = new StringBuilder();
        for (StringBuilder row : rows) {
            result.append(row);
        }

        return result.toString();
    }

    public int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            int pop = x % 10;
            x /= 10;
            if (rev > Integer.MAX_VALUE / 10 || (rev == Integer.MAX_VALUE / 10 && pop > 7)) {
                return 0;
            }
            if (rev < Integer.MIN_VALUE / 10 || (rev == Integer.MIN_VALUE / 10 && pop < -8)) {
                return 0;
            }
            rev = rev * 10 + pop;
        }
        return rev;
    }

    public int myAtoi(String str) {
        str = str.trim();
        long result = 0;
        if (str.isEmpty()) {
            return 0;
        }
        int sign = 1;
        if (str.charAt(0) == '-') {
            sign = -1;
            str = str.substring(1);
        } else if (str.charAt(0) == '+') {
            str = str.substring(1);
        }

        for (char c : str.toCharArray()) {
            if (c >= '0' && c <= '9') {
                result = result * 10 + c - '0';
            } else {
                break;
            }

            if (result > Integer.MAX_VALUE && sign == 1) {
                return Integer.MAX_VALUE;
            } else if (result > (long) Integer.MAX_VALUE + 1 && sign == -1) {
                return Integer.MIN_VALUE;
            }
        }

        result = sign * result;
        return (int) result;
    }

    public boolean isPalindrome(int x) {
        long origin = x;
        if (x < 0) {
            return false;
        }
        long result = 0;
        while (x > 0) {
            result = result * 10 + x % 10;
            x /= 10;
        }
        return result == origin;
    }

    public boolean isMatch(String s, String p) {
        if (p.equals("")) {
            return s.equals("");
        }

        boolean first_match = s.length() > 0 && (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.');

        if (p.length() >= 2 && p.charAt(1) == '*') {
            return isMatch(s, p.substring(2)) || (first_match && isMatch(s.substring(1), p));
        } else {
            return first_match && isMatch(s.substring(1), p.substring(1));
        }
    }

    enum Result {
        TRUE,
        FALSE
    }

    Result[][] memo;

    public boolean isMatch2(String text, String pattern) {
        memo = new Result[text.length() + 1][pattern.length() + 1];
        return isMatch2Dp(0, 0, text, pattern);
    }

    public boolean isMatch2Dp(int i, int j, String text, String pattern) {
        if (memo[i][j] != null) {
            return memo[i][j] == Result.TRUE;
        }
        boolean ans;
        if (j == pattern.length()) {
            ans = i == text.length();
        } else {
            boolean first_match =
                    (i < text.length() && (pattern.charAt(j) == text.charAt(i) || pattern.charAt(j) == '.'));
            if (j + 1 < pattern.length() && pattern.charAt(j + 1) == '*') {
                ans =
                        (isMatch2Dp(i, j + 2, text, pattern))
                                || first_match && isMatch2Dp(i + 1, j, text, pattern);
            } else {
                ans = first_match && isMatch2Dp(i + 1, j + 1, text, pattern);
            }
        }
        memo[i][j] = ans ? Result.TRUE : Result.FALSE;
        return ans;
    }

    public TreeNode recoverFromPreorder(String S) {
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        Queue<Integer> depthQueue = new LinkedList<>();
        generateNodeQueues(S, nodeQueue, depthQueue);
        if (nodeQueue.size() == 0) {
            return null;
        }
        Stack<TreeNode> nodeStack = new Stack<>();
        Stack<Integer> depthStack = new Stack<>();

        TreeNode root = nodeQueue.remove();
        int depth = depthQueue.remove();
        nodeStack.push(root);
        depthStack.push(depth);

        while (nodeQueue.size() > 0) {
            TreeNode node = nodeQueue.remove();
            depth = depthQueue.remove();

            TreeNode parent = null;
            int parentDepth = -1;
            while (true) {
                parent = nodeStack.pop();
                parentDepth = depthStack.pop();
                if (parentDepth == depth - 1) {
                    break;
                }
            }
            if (parent.left == null) {
                parent.left = node;
                nodeStack.push(parent);
                depthStack.push(parentDepth);
            } else {
                parent.right = node;
            }
            nodeStack.push(node);
            depthStack.push(depth);
        }
        return root;
    }

    private void generateNodeQueues(String S, Queue<TreeNode> nodeQueue, Queue<Integer> depthQueue) {
        if (S.length() == 0) {
            return;
        }
        boolean isDigit = false;
        int depth = 0;
        int value = 0;
        for (int i = 0; i < S.length(); i++) {
            if ('-' != S.charAt(i)) {
                if (isDigit) {
                    value = value * 10 + S.charAt(i) - '0';
                } else {
                    isDigit = true;
                    value = S.charAt(i) - '0';
                }
            } else {
                if (!isDigit) {
                    depth++;
                } else {
                    nodeQueue.add(new TreeNode(value));
                    depthQueue.add(depth);
                    isDigit = false;
                    depth = 1;
                }
            }
        }
        nodeQueue.add(new TreeNode(value));
        depthQueue.add(depth);
    }

    public int maxProfit(int k, int[] prices) {
        if (k == 0 || prices.length == 0)
            return 0;

        if (2*k>prices.length){
            int result = 0;
            for(int i = 0; i<prices.length-1; i++){
                if(prices[i+1]>prices[i]){
                    result += prices[i+1]-prices[i];
                }
            }
            return result;
        }

        int[][][] dp = new int[prices.length+1][k+1][2];

        dp[0][0][1] = Integer.MIN_VALUE;

        for(int i = 1; i < prices.length+1; i++){
            dp[i][0][1] = Math.max(dp[i-1][0][1], -prices[i-1]);
        }

        for(int i = 1; i <= k ; i++){
            dp[0][i][1] = -prices[0];
        }

        for(int i = 1; i < prices.length+1; i++){
            for(int j = 1; j<=k; j++){
                dp[i][j][0] = Math.max(dp[i-1][j][0],dp[i-1][j-1][1]+prices[i-1]);
                dp[i][j][1] = Math.max(dp[i-1][j][1],dp[i-1][j][0]-prices[i-1]);
            }
        }

        return dp[prices.length][k][0];
    }

    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length -1;
        int result = 0;

        while(left<right){
            int minH = -1;
            int wide = right - left;
            if (height[left]<=height[right]){
                minH = height[left];
                left++;
            }else {
                minH = height[right];
                right--;
            }
            result = Math.max(result, wide*minH);
        }
        return result;
    }
}
