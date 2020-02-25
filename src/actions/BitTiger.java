package actions;

import actions.util.ListNode;

import java.util.HashMap;

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
        while (l1 != null && l2 != null) {
            int value = l1.val + l2.val + carry;
            carry = value / 10; //BUG: / and %
            value = value % 10;
            ListNode newNode = new ListNode(value);
            prev.next = newNode;
            prev = newNode;
            l1 = l1.next;
            l2 = l2.next;
        }
        if (l1 != null) {
            prev.next = l1;
        } else {
            prev.next = l2;
        }
        return dummy.next;

    }
}
