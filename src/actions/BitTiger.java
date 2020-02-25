package actions;

import actions.util.ListNode;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

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
}
