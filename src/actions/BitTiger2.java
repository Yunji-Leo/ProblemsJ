package actions;

import actions.util.ListNode;

import java.util.*;

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

    public int calculateMinimumHP(int[][] dungeon) {
        if (dungeon == null || dungeon.length == 0) {
            return 0;
        }
        int M = dungeon.length;
        int N = dungeon[0].length;
        int[][] dp = new int[M][N];
        dp[M - 1][N - 1] = Math.max(1, 1 - dungeon[M - 1][N - 1]);
        for (int i = M - 2; i >= 0; i--) {
            dp[i][N - 1] = Math.max(1, dp[i + 1][N - 1] - dungeon[i][N - 1]);
        }
        for (int j = N - 2; j >= 0; j--) {
            dp[M - 1][j] = Math.max(1, dp[M - 1][j + 1] - dungeon[M - 1][j]);
        }

        for (int i = M - 2; i >= 0; i--) {
            for (int j = N - 2; j >= 0; j--) {
                dp[i][j] = Math.max(1, Math.min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]);
            }
        }

        return dp[0][0];
    }

    public String largestNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();

        String[] snums = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            snums[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(snums, new LargestNumberComparator());
        for (String s : snums) {
            sb.append(s);
        }
        if (sb.charAt(0) == '0')
            return "0";
        return sb.toString();
    }

    class LargestNumberComparator implements Comparator<String> {
        @Override
        public int compare(String s1, String s2) {
            int i = 0;
            while (i < s1.length() && i < s2.length()) {
                if (s1.charAt(i) < s2.charAt(i)) {
                    return 1;
                } else if (s1.charAt(i) > s2.charAt(i)) {
                    return -1;
                }
                i++;
            }
            if (s1.length() == s2.length()) {
                return 0;
            }
            if (s1.length() > s2.length()) {
                for (int k = s1.length() - 1; k > 0; k--) {
                    if (s1.charAt(k) > s1.charAt(0)) {
                        return -1;
                    } else if (s1.charAt(k) < s1.charAt(0)) {
                        return 1;
                    } else if (s1.charAt(k) > s1.charAt(k - 1)) {
                        return -1;
                    } else if (s1.charAt(k) < s1.charAt(k - 1)) {
                        return 1;
                    }
                }
                return 0;
            } else {
                for (int k = s2.length() - 1; k > 0; k--) {
                    if (s2.charAt(k) > s2.charAt(0)) {
                        return 1;
                    } else if (s2.charAt(k) < s2.charAt(0)) {
                        return -1;
                    } else if (s2.charAt(k) > s2.charAt(k - 1)) {
                        return 1;
                    } else if (s2.charAt(k) < s2.charAt(k - 1)) {
                        return -1;
                    }
                }
                return 0;
            }
        }
    }

    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        if (nums.length == 2) {
            return nums[0] > nums[1] ? nums[0] : nums[1];
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = nums[0] > nums[1] ? nums[0] : nums[1];

        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }

        return dp[nums.length - 1];
    }


    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int count = 0;
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    DFSMarking(grid, i, j, m, n);
                    count++;
                }
            }
        }
        return count;
    }

    private void DFSMarking(char[][] grid, int i, int j, int M, int N) {
        if (i < 0 || j < 0 || i >= M || j >= N || grid[i][j] != '1') {
            return;
        }

        grid[i][j] = '0';
        DFSMarking(grid, i + 1, j, M, N);
        DFSMarking(grid, i - 1, j, M, N);
        DFSMarking(grid, i, j + 1, M, N);
        DFSMarking(grid, i, j - 1, M, N);
    }

    public boolean isHappy(int n) {
        HashSet<Integer> set = new HashSet<>();
        while (n != 1 && !set.contains(n)) {
            set.add(n);
            int result = 0;
            while (n != 0) {
                result += (n % 10) * (n % 10);
                n /= 10;
            }
            n = result;
        }
        return n == 1;
    }

    public int countPrimes(int n) {
        int count = 0;
        boolean[] nonPrime = new boolean[n];
        for (int i = 2; i < n; i++) {
            if (!nonPrime[i]) {
                count++;
                int x = i;
                while (x < n) {
                    nonPrime[x] = true;
                    x += i;
                }
            }
        }
        return count;
    }

}
