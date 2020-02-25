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
}
