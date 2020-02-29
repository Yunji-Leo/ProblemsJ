package actions;

import actions.util.ListNode;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;

public class BitTiger2 {
    class MinStack {
        Deque<Integer> stack;
        Deque<Integer> min;

        /**
         * initialize your data structure here.
         */
        public MinStack() {
            stack = new ArrayDeque<>();
            min = new ArrayDeque<>();
        }

        public void push(int x) {
            stack.push(x);
            if (!min.isEmpty() && min.peek() < x) {
                min.push(min.peek());
            } else {
                min.push(x);
            }
        }

        public void pop() {
            stack.pop();
            min.pop();
        }

        public int top() {
            return stack.peek();
        }

        public int getMin() {
            return min.peek();
        }
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null)
            return null;
        ListNode runnerA = headA;
        ListNode runnerB = headB;
        boolean flip = false;
        while (true) {
            if (runnerA == runnerB) {
                return runnerA;
            }
            if (runnerA == null) {
                if (!flip) {
                    runnerA = headB;
                    flip = true;
                } else {
                    return null;
                }
            }
            if (runnerB == null) {
                runnerB = headA;
            }
            runnerA = runnerA.next;
            runnerB = runnerB.next;
        }
    }

    public int findPeakElement(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (nums[mid] > nums[mid + 1])
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }

    public String fractionToDecimal(int numeratorInt, int denominatorInt) {
        long numerator = (long) numeratorInt;
        long denominator = (long) denominatorInt;
        long quotient = numerator / denominator;
        long remainder = numerator % denominator;
        String result = String.valueOf(quotient);

        if (remainder == 0) {
            return result;
        }
        if (quotient == 0 && (numerator > 0 && denominator < 0 || numerator < 0 && denominator > 0)) {
            result = "-" + result;
        }
        result += ".";

        numerator = Math.abs(remainder);
        denominator = Math.abs(denominator);
        HashMap<Long, Integer> recurIndexMap = new HashMap<>();
        StringBuilder deciResult = new StringBuilder();
        while (true) {
            if (!recurIndexMap.containsKey(numerator)) {
                recurIndexMap.put(numerator, deciResult.length());
                int preZero = -1;
                while (numerator < denominator) {
                    preZero++;
                    numerator *= 10;
                }
                while (preZero > 0) {
                    deciResult.append("0");
                    preZero--;
                }
                quotient = numerator / denominator;
                remainder = numerator % denominator;
                String qValue = String.valueOf(quotient);
                deciResult.append(qValue);
                if (remainder == 0) {
                    break;
                }
                numerator = remainder;
            } else {
                int index = recurIndexMap.get(numerator);
                String firstPart = deciResult.substring(0, index);
                String secondPart = deciResult.substring(index, deciResult.length());
                deciResult = new StringBuilder(firstPart + "(" + secondPart + ")");
                break;
            }
        }
        return result + deciResult;
    }

    public int majorityElement(int[] nums) {
        int result = nums[0];
        int count = 1;

        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != result) {
                if (count == 0) {
                    result = nums[i];
                    count = 1;
                } else {
                    count--;
                }
            } else {
                count++;
            }
        }

        return result;
    }

    public int titleToNumber(String s) {
        int result = 0;
        int multiplier = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            int value = s.charAt(i) - 'A' + 1;
            int mul = 1;
            for (int j = 0; j < multiplier; j++) {
                mul *= 26;
            }
            result += mul * value;
            multiplier++;
        }
        return result;
    }

    public int trailingZeroes(int n) {
        int result = 0;
        while (n / 5 != 0) {
            result += n / 5;
            n /= 5;
        }
        return result;
    }
}
