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

        if (2 * k > prices.length) {
            int result = 0;
            for (int i = 0; i < prices.length - 1; i++) {
                if (prices[i + 1] > prices[i]) {
                    result += prices[i + 1] - prices[i];
                }
            }
            return result;
        }

        int[][][] dp = new int[prices.length + 1][k + 1][2];

        dp[0][0][1] = Integer.MIN_VALUE;

        for (int i = 1; i < prices.length + 1; i++) {
            dp[i][0][1] = Math.max(dp[i - 1][0][1], -prices[i - 1]);
        }

        for (int i = 1; i <= k; i++) {
            dp[0][i][1] = -prices[0];
        }

        for (int i = 1; i < prices.length + 1; i++) {
            for (int j = 1; j <= k; j++) {
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j - 1][1] + prices[i - 1]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j][0] - prices[i - 1]);
            }
        }

        return dp[prices.length][k][0];
    }

    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int result = 0;

        while (left < right) {
            int minH = -1;
            int wide = right - left;
            if (height[left] <= height[right]) {
                minH = height[left];
                left++;
            } else {
                minH = height[right];
                right--;
            }
            result = Math.max(result, wide * minH);
        }
        return result;
    }

    public String intToRoman(int num) {
        String[] M = {"", "M", "MM", "MMM"};
        String[] C = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String[] X = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String[] I = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        return M[num / 1000] + C[(num % 1000) / 100] + X[(num % 100) / 10] + I[num % 10];

    }

    public int romanToInt(String s) {
        HashMap<Character, Integer> roman = new HashMap<>();
        roman.put('M', 1000);
        roman.put('D', 500);
        roman.put('C', 100);
        roman.put('L', 50);
        roman.put('X', 10);
        roman.put('V', 5);
        roman.put('I', 1);
        int result = 0;
        for (int i = 0; i < s.length() - 1; i++) {
            if (roman.get(s.charAt(i)) < roman.get(s.charAt(i + 1))) {
                result -= roman.get(s.charAt(i));
            } else {
                result += roman.get(s.charAt(i));
            }
        }
        return result + roman.get(s.charAt(s.length() - 1));
    }

    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0)
            return "";
        return longestCommonPrefix(strs, 0, strs.length - 1);
    }

    private String longestCommonPrefix(String[] strs, int l, int r) {
        if (l == r) {
            return strs[l];
        } else {
            int mid = (l + r) / 2;
            String lcpLeft = longestCommonPrefix(strs, l, mid);
            String lcpRight = longestCommonPrefix(strs, mid + 1, r);
            return commonPrefix(lcpLeft, lcpRight);
        }
    }

    private String commonPrefix(String left, String right) {
        int min = Math.min(left.length(), right.length());
        for (int i = 0; i < min; i++) {
            if (left.charAt(i) != right.charAt(i))
                return left.substring(0, i);
        }
        return left.substring(0, min);
    }

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < nums.length - 2; i++) {
            if (i == 0 || nums[i] != nums[i - 1]) {
                int left = i + 1;
                int right = nums.length - 1;
                while (left < right) {
                    int sum = nums[i] + nums[left] + nums[right];
                    if (left != i + 1 && nums[left] == nums[left - 1]) {
                        left++;
                    } else if (right != nums.length - 1 && nums[right] == nums[right + 1]) {
                        right--;
                    } else if (sum < 0) {
                        left++;
                    } else if (sum > 0) {
                        right--;
                    } else {
                        List<Integer> ans = new ArrayList<>();
                        ans.add(nums[i]);
                        ans.add(nums[left]);
                        ans.add(nums[right]);
                        result.add(ans);
                        left++;
                        right--;
                    }
                }
            }
        }
        return result;
    }

    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int result = Integer.MAX_VALUE;
        int distance = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length - 2; i++) {
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (Math.abs(sum - target) < distance) {
                    result = sum;
                    distance = Math.abs(sum - target);
                }
                if (sum == target)
                    return sum;
                if (sum > target) {
                    right--;
                } else {
                    left++;
                }
            }
        }
        return result;
    }

    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();

        if (digits == null || digits.length() == 0) {
            return result;
        }

        HashMap<Character, String[]> phone = new HashMap<>();
        phone.put('2', new String[]{"a", "b", "c"});
        phone.put('3', new String[]{"d", "e", "f"});
        phone.put('4', new String[]{"g", "h", "i"});
        phone.put('5', new String[]{"j", "k", "l"});
        phone.put('6', new String[]{"m", "n", "o"});
        phone.put('7', new String[]{"p", "q", "r", "s"});
        phone.put('8', new String[]{"t", "u", "v"});
        phone.put('9', new String[]{"w", "x", "y", "z"});

        letterCombinationsRecursive(result, phone, digits, 0, "");
        return result;
    }

    private void letterCombinationsRecursive(List<String> result, HashMap<Character, String[]> phone, String digits, int pos, String temp) {
        if (pos == digits.length()) {
            result.add(temp);
            return;
        }

        for (String s : phone.get(digits.charAt(pos))) {
            temp += s;
            letterCombinationsRecursive(result, phone, digits, pos + 1, temp);
            temp = temp.substring(0, temp.length() - 1);
        }
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 3; i++) {
            if (i == 0 || nums[i] != nums[i - 1]) {
                for (int j = i + 1; j < nums.length - 2; j++) {
                    if (j == i + 1 || nums[j] != nums[j - 1]) {
                        int left = j + 1;
                        int right = nums.length - 1;
                        while (left < right) {
                            int sum = nums[i] + nums[j] + nums[left] + nums[right];
                            if (sum == target) {
                                List<Integer> ans = new ArrayList<>();
                                ans.add(nums[i]);
                                ans.add(nums[j]);
                                ans.add(nums[left]);
                                ans.add(nums[right]);
                                result.add(ans);

                                while (left < right && nums[left] == nums[left + 1]) {
                                    left++;
                                }
                                while (left < right && nums[right] == nums[right - 1]) {
                                    right--;
                                }
                                left++;
                                right--;
                            } else if (sum > target) {
                                right--;
                            } else {
                                left++;
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode first = dummy;
        for (int i = 0; i < n; i++) {
            first = first.next;
        }
        ListNode second = dummy;
        while (first.next != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return dummy.next;
    }

    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (Character c : s.toCharArray()) {
            if (c == '(') {
                stack.push(')');
            } else if (c == '[') {
                stack.push(']');
            } else if (c == '{') {
                stack.push('}');
            } else if (stack.size() == 0 || c != stack.peek()) {
                return false;
            } else {
                stack.pop();
            }
        }
        return stack.size() == 0;
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val > l2.val) {
                prev.next = l2;
                l2 = l2.next;
            } else {
                prev.next = l1;
                l1 = l1.next;
            }
            prev = prev.next;
        }
        if (l1 != null) {
            prev.next = l1;
        } else {
            prev.next = l2;
        }
        return dummy.next;
    }

    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        generateParenthesisRecursive(result, "", 0, 0, n);
        return result;
    }

    private void generateParenthesisRecursive(List<String> result, String temp, int left, int right, int n) {
        if (temp.length() == 2 * n) {
            result.add(temp);
            return;
        }
        if (left < n) {
            generateParenthesisRecursive(result, temp + "(", left + 1, right, n);
        }
        if (right < left) {
            generateParenthesisRecursive(result, temp + ")", left, right + 1, n);
        }
    }

    public ListNode mergeKLists(List<ListNode> lists) {
        if (lists == null || lists.size() == 0)
            return null;

        PriorityQueue<ListNode> queue = new PriorityQueue<>(lists.size(), new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                if (o1.val < o2.val) {
                    return -1;
                } else if (o1.val == o2.val) {
                    return 0;
                } else {
                    return 1;
                }
            }
        });

        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;

        for (ListNode node : lists) {
            if (node != null) {
                queue.add(node);
            }
        }

        while (!queue.isEmpty()) {
            tail.next = queue.poll();
            tail = tail.next;

            if (tail.next != null) {
                queue.add(tail.next);
            }
        }
        return dummy.next;
    }

    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        prev.next = head;
        while (prev.next != null && prev.next.next != null) {
            ListNode first = prev.next;
            ListNode second = prev.next.next;
            prev.next = second;
            first.next = second.next;
            second.next = first;
            prev = first;
        }
        return dummy.next;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        Stack<ListNode> stack = new Stack<>();
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        prev.next = head;
        outerloop:
        while (prev.next != null) {
            ListNode cur = prev.next;
            for (int i = 0; i < k; i++) {
                stack.push(cur);
                cur = cur.next;
                if (cur == null && i < k - 1) {
                    break outerloop;
                }
            }
            while (stack.size() > 0) {
                ListNode rev = stack.pop();
                prev.next = rev;
                prev = rev;
            }
            prev.next = cur;
        }
        return dummy.next;
    }

    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode tmpRight = null;
        if (root.right != null) {
            flatten(root.right);
            tmpRight = root.right;
            root.right = null;
        }
        if (root.left != null) {
            flatten(root.left);
            root.right = root.left;
            root.left = null;
        }
        TreeNode curr = root;
        while (curr.right != null) {
            curr = curr.right;
        }
        curr.right = tmpRight;
    }


    public void flatten2(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode tmpRight = null;
        if (root.left != null) {
            flatten2(root.left);
            tmpRight = root.right;
            root.right = root.left;
            root.left = null;
        } else {
            tmpRight = root.right;
            root.right = null;
        }

        if (tmpRight != null) {
            flatten2(tmpRight);
        }

        TreeNode curr = root;
        while (curr.right != null) {
            curr = curr.right;
        }
        curr.right = tmpRight;
    }

    public void flatten3(TreeNode root){
        if (root == null){
            return;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);

        while(stack.size()>0){
            TreeNode curr = stack.pop();
            if (curr.right!=null){
                stack.push(curr.right);
            }
            if (curr.left!=null){
                stack.push(curr.left);
            }
            curr.left = null;
            queue.add(curr);
        }

        TreeNode curr = queue.poll();
        while (queue.size()>0){
            curr.right = queue.poll();
            curr = curr.right;
        }
    }

}
