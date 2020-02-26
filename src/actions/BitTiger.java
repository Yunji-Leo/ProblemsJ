package actions;

import actions.util.ListNode;

import java.util.*;

public class BitTiger {

    public int[] twoSum(int[] nums, int target) {
        int[] ans = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                return new int[]{map.get(nums[i]), i};
            }
            map.put(target - nums[i], i);
        }
        return ans;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int value = carry;
            if (l1 != null) {
                value += l1.val;
            }
            if (l2 != null) {
                value += l2.val;
            }
            carry = value / 10; //BUG: / and %
            value = value % 10;
            ListNode newNode = new ListNode(value);
            prev.next = newNode;
            prev = newNode;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry != 0) {
            prev.next = new ListNode(carry);
        }
        return dummy.next;
    }

    public int lengthOfLongestSubstring(String s) {
        int fast = 0;
        int slow = 0;
        int result = 0;
        HashSet<Character> hset = new HashSet<>();
        while (fast < s.length()) {
            if (hset.contains(s.charAt(fast))) {
                while (s.charAt(slow) != s.charAt(fast)) {
                    hset.remove(s.charAt(slow));
                    slow++;
                }
                hset.remove(s.charAt(slow));
                slow++;
            }
            hset.add(s.charAt(fast));
            fast++;
            result = Math.max(result, fast - slow);
        }
        return result;
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if (m + n % 2 == 0) {
            return (double) (getKth(nums1, nums2, (m + n) / 2 + 1) + getKth(nums1, nums2, (m + n) / 2)) * 0.5;
        } else {
            return (double) getKth(nums1, nums2, (m + n) / 2) * 1.0;
        }
    }

    private int getKth(int[] A, int[] B, int k) {
        int m = A.length;
        int n = B.length;

        if (m > n) {
            return getKth(B, A, k);
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
            return getKth(Arrays.copyOfRange(A, pa, m), B, pb);
        } else {
            return getKth(A, Arrays.copyOfRange(B, pb, n), pa);
        }
    }

    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }

        int start = 0;
        int end = 0;

        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAtCenter(s, i, i);
            int len2 = expandAtCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start + 1) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }

        }
        return s.substring(start, end + 1);
    }

    private int expandAtCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length()) {
            if (s.charAt(left) == s.charAt(right)) {
                left--;
                right++;
            } else {
                break;
            }
        }
        return right - left - 1;
    }

    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int result = 0;
        while (left < right) {
            if (height[left] <= height[right]) {
                result = Math.max(result, height[left] * (right - left));
                left++;
            } else {
                result = Math.max(result, height[right] * (right - left));
                right--;
            }
        }
        return result;
    }

    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }

        for (int i = 0; i < strs[0].length(); i++) {
            char c = strs[0].charAt(i);
            for (int j = 1; j < strs.length; j++) {
                if (i == strs[j].length() || strs[j].charAt(i) != c) {
                    return strs[0].substring(0, i);
                }
            }
        }
        return strs[0];
    }

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        if (nums.length < 3) {
            return results;
        }
        Arrays.sort(nums);
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
                        List<Integer> res = new ArrayList<>();
                        res.add(nums[i]);
                        res.add(nums[left]);
                        res.add(nums[right]);
                        results.add(res);
                        left++;
                        right--;
                    }
                }
            }
        }
        return results;
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

        letterCombinationsBacktrack(result, phone, digits, 0, "");
        return result;
    }

    private void letterCombinationsBacktrack(List<String> result, HashMap<Character, String[]> phone, String digits, int pos, String tmp) {
        if (pos == digits.length()) {
            result.add(tmp);
            return;
        }

        for (String s : phone.get(digits.charAt(pos))) {
            tmp += s;
            letterCombinationsBacktrack(result, phone, digits, pos + 1, tmp);
            tmp = tmp.substring(0, tmp.length() - 1);
        }
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
            } else if (stack.empty() || c != stack.peek()) {
                return false;
            } else {
                stack.pop();
            }
        }
        return stack.empty();
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                prev.next = l1;
                prev = l1;
                l1 = l1.next;
            } else {
                prev.next = l2;
                prev = l2;
                l2 = l2.next;
            }
        }
        if (l1 == null) {
            prev.next = l2;
        } else {
            prev.next = l1;
        }
        return dummy.next;
    }

    public List<String> generateParenthesis(int n) {
        List<String> results = new ArrayList<>();
        generateParenthesisBacktrack(results, n, n, n, "");
        return results;
    }

    private void generateParenthesisBacktrack(List<String> results, int left, int right, int n, String temp) {
        if (left == n && right == n) {
            results.add(temp);
            return;
        }

        if (right > left || left > n) {
            return;
        }
        generateParenthesisBacktrack(results, left + 1, right, n, temp + "(");
        generateParenthesisBacktrack(results, left, right + 1, n, temp + ")");
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        return mergeKListsDC(lists, 0, lists.length - 1);
    }

    private ListNode mergeKListsDC(ListNode[] lists, int left, int right) {
        if (left == right) {
            return lists[left];
        }
        int mid = (left + right) / 2;
        ListNode leftList = mergeKListsDC(lists, left, mid);
        ListNode rightList = mergeKListsDC(lists, mid + 1, right);
        return mergeTwoLists(leftList, rightList);
    }

    public int divide(int dividend, int divisor) {
        return 0;
    }

    private long divideRecursive(long dividend, long divisor) {
        if (dividend < divisor) {
            return 0;
        }
        long sum = divisor;
        long multiplier = 1;
        while (sum <= dividend) {
            sum += sum;
            multiplier += multiplier;
        }
        return multiplier / 2 + divideRecursive(dividend - sum / 2, divisor);
    }

    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]) {
                j--;
            }
            int tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
        }
        reverse(nums, i + 1, nums.length - 1);
    }

    private void reverse(int[] nums, int left, int right) {
        while (left < right) {
            int tmp = nums[left];
            nums[left] = nums[right];
            nums[right] = tmp;
            left++;
            right--;
        }
    }

    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        if (nums.length == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int k = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] < nums[i - 1]) {
                k = i;
            }
        }
        int left = k;
        int right = k + nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int realMid = mid;
            if (mid >= nums.length) {
                realMid = mid - nums.length; //+ 1;
            }
            if (nums[realMid] == target) {
                return realMid;
            } else if (nums[realMid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    public String countAndSay(int n) {
        String s = "1";
        while (n > 1) {
            int num = 0;
            StringBuilder ans = new StringBuilder();
            for (int i = 0; i < s.length(); i++) {
                if (i == 0 || s.charAt(i) == s.charAt(i - 1)) {
                    num++;
                } else {
                    ans.append(num).append(s.charAt(i - 1));
                    num = 1;
                }
            }
            ans.append(num).append(s.charAt(s.length() - 1));
            s = ans.toString();
            n--;
        }
        return s;
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> results = new ArrayList<>();
        Arrays.sort(candidates);
        combinationSumBacktrack(candidates, 0, results, new ArrayList<>(), 0, target);
        return results;
    }

    private void combinationSumBacktrack(int[] candidates, int index, List<List<Integer>> results, List<Integer> temp, int sum, int target) {
        if (sum == target) {
            results.add(new ArrayList<>(temp));
            return;
        }

        for (int i = index; i < candidates.length; i++) {
            if (sum + candidates[i] > target) {
                return;
            }
            temp.add(candidates[i]);
            combinationSumBacktrack(candidates, i, results, temp, sum + candidates[i], target);
            temp.remove(temp.size() - 1);
        }
    }

    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 1;
        }

        for (int i = 0; i < nums.length; i++) {
            while (nums[i] - 1 != i && nums[i] >= 1 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            }
        }

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }

        return nums.length + 1;
        return 1;
    }

    private void swap(int[] nums, int left, int right) {
        int tmp = nums[left];
        nums[left] = nums[right];
        nums[right] = tmp;
    }

    public int minDistance(String word1, String word2) {
        if (word1 == null && word2 == null) {
            return 0;
        }
        if (word1 == "") {
            return word2.length();
        }
        if (word2 == "") {
            return word1.length();
        }

        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 0; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= word2.length(); j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {
                if (word1.charAt(i - 1) != word2.charAt(j - 1)) {
                    dp[i][j] = Math.min(dp[i - 1][j - 1] + 1, Math.min(dp[i][j - 1] + 1, dp[i - 1][j] + 1));
                } else {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }

        return dp[word1.length()][word2.length()];
    }
}
