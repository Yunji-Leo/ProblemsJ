package actions;

import actions.util.ListNode;
import actions.util.TreeNode;

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

    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        HashMap<Integer, List<Integer>> prereqMap = new HashMap<>();
        HashSet<Integer> canFinishCourses = new HashSet<>();
        for (int i = 0; i < prerequisites.length; i++) {
            if (!prereqMap.containsKey(prerequisites[i][0])) {
                prereqMap.put(prerequisites[i][0], new ArrayList<>());
            }
            prereqMap.get(prerequisites[i][0]).add(prerequisites[i][1]);
        }
        for (int i = 0; i < numCourses; i++) {
            if (!canFinishRecur(i, prereqMap, canFinishCourses, new HashSet<>())) {
                return false;
            }
        }
        return true;
    }

    private boolean canFinishRecur(int course, HashMap<Integer, List<Integer>> prereqMap, HashSet<Integer> canFinishCourses, HashSet<Integer> visited) {
        if (canFinishCourses.contains(course)) {
            return true;
        }
        if (visited.contains(course)) {
            return false;
        }
        visited.add(course);
        if (!prereqMap.containsKey(course)) {
            canFinishCourses.add(course);
            return true;
        }
        for (int preCourse : prereqMap.get(course)) {
            if (!canFinishRecur(preCourse, prereqMap, canFinishCourses, visited)) {
                return false;
            }
        }
        canFinishCourses.add(course);
        return true;
    }

    class Trie {

        TrieNode root;

        /**
         * Initialize your data structure here.
         */
        public Trie() {
            root = new TrieNode();
        }

        /**
         * Inserts a word into the trie.
         */
        public void insert(String word) {
            TrieNode cur = root;
            for (char c : word.toCharArray()) {
                if (cur.nodes[c - 'a'] == null) {
                    cur.nodes[c - 'a'] = new TrieNode();
                }
                cur = cur.nodes[c - 'a'];
            }
            cur.isWord = true;
        }

        /**
         * Returns if the word is in the trie.
         */
        public boolean search(String word) {
            TrieNode cur = root;
            for (char c : word.toCharArray()) {
                if (cur.nodes[c - 'a'] == null) {
                    return false;
                }
                cur = cur.nodes[c - 'a'];
            }
            return cur.isWord == true;
        }

        /**
         * Returns if there is any word in the trie that starts with the given prefix.
         */
        public boolean startsWith(String prefix) {
            TrieNode cur = root;
            for (char c : prefix.toCharArray()) {
                if (cur.nodes[c - 'a'] == null) {
                    return false;
                }
                cur = cur.nodes[c - 'a'];
            }
            return true;
        }

        class TrieNode {
            boolean isWord;
            TrieNode[] nodes;

            public TrieNode() {
                nodes = new TrieNode[26];
            }
        }
    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses == 0) {
            return new int[]{};
        }
        int[] result = new int[numCourses];
        HashMap<Integer, List<Integer>> reverseDeps = new HashMap<>();
        int[] degrees = new int[numCourses];
        for (int[] prereq : prerequisites) {
            degrees[prereq[0]]++;
            if (!reverseDeps.containsKey(prereq[1])) {
                reverseDeps.put(prereq[1], new ArrayList<>());
            }
            reverseDeps.get(prereq[1]).add(prereq[0]);
        }

        Queue<Integer> orderQueue = new LinkedList<>();
        int index = 0;
        for (int i = 0; i < numCourses; i++) {
            if (degrees[i] == 0) {
                orderQueue.add(i);
            }
        }

        while (!orderQueue.isEmpty()) {
            int course = orderQueue.poll();
            result[index] = course;
            index++;
            if (reverseDeps.containsKey(course)) {
                for (int dep : reverseDeps.get(course)) {
                    degrees[dep]--;
                    if (degrees[dep] == 0) {
                        orderQueue.add(dep);
                    }
                }
            }
        }
        if (index != numCourses) {
            return new int[]{};
        }
        return result;
    }

    public int rob2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        if (nums.length == 2) {
            return nums[0] > nums[1] ? nums[0] : nums[1];
        }
        if (nums.length == 3) {
            return Math.max(nums[0], Math.max(nums[1], nums[2]));
        }

        int[] dp = new int[nums.length - 1];
        dp[0] = nums[0];
        dp[1] = nums[0] > nums[1] ? nums[0] : nums[1];
        for (int i = 2; i < nums.length - 1; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        int result = dp[dp.length - 1];
        dp[0] = nums[1];
        dp[1] = nums[1] > nums[2] ? nums[1] : nums[2];
        for (int i = 3; i < nums.length; i++) {
            dp[i - 1] = Math.max(dp[i - 1 - 1], dp[i - 1 - 2] + nums[i]);
        }
        return Math.max(result, dp[dp.length - 1]);
    }

    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k == 0) {
            return 0;
        }
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int n : nums) {
            if (k > 0) {
                minHeap.add(n);
                k--;
            } else if (n > minHeap.peek()) {
                minHeap.add(n);
                minHeap.poll();
            }
        }
        return minHeap.poll();
    }

    public boolean containsDuplicate(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for (int n : nums) {
            if (set.contains(n)) {
                return true;
            }
            set.add(n);
        }
        return false;
    }

    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<List<Integer>> result = new ArrayList<>();
        if (buildings == null || buildings.length == 0) {
            return result;
        }

        PriorityQueue<int[]> buildingQueue = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] != o2[0]) {
                    return o1[0] - o2[0];
                }
                return o2[1] - o1[1];
            }
        });

        for (int[] b : buildings) {
            buildingQueue.add(new int[]{b[0], b[2]});
            buildingQueue.add(new int[]{b[1], -b[2]});
        }

        PriorityQueue<Integer> heightQueue = new PriorityQueue<>(Collections.reverseOrder());
        heightQueue.add(0);
        int curHeight = 0;
        while (!buildingQueue.isEmpty()) {
            int[] b = buildingQueue.poll();
            if (b[1] > 0) {
                if (b[1] > curHeight) {
                    List<Integer> l = new ArrayList<>();
                    l.add(b[0]);
                    l.add(b[1]);
                    result.add(l);
                    curHeight = b[1];
                }
                heightQueue.add(b[1]);
            } else {
                heightQueue.remove(-b[1]);
                if (curHeight > heightQueue.peek()) {
                    curHeight = heightQueue.peek();
                    List<Integer> l = new ArrayList<>();
                    l.add(b[0]);
                    l.add(curHeight);
                    result.add(l);
                }
            }
        }
        return result;
    }

    public int calculate(String s) {
        Stack<Integer> stack = new Stack<>();
        int operand = 0;
        int result = 0;
        int sign = 1;

        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (Character.isDigit(ch)) {
                operand = 10 * operand + (int) (ch - '0');
            } else if (ch == '+') {
                result += sign * operand;
                sign = 1;
                operand = 0;
            } else if (ch == '-') {
                result += sign * operand;
                sign = -1;
                operand = 0;
            } else if (ch == '(') {
                stack.push(result);
                stack.push(sign);
                sign = 1;
                result = 0;
            } else if (ch == ')') {
                result += sign * operand;
                result *= stack.pop();
                result += stack.pop();
                operand = 0;
            }
        }

        return result + (sign * operand);
    }

    public int calculate2(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int len = s.length();
        Stack<Integer> stack = new Stack<>();
        int num = 0;
        char sign = '+';
        for (int i = 0; i < len; i++) {
            if (Character.isDigit(s.charAt(i))) {
                num = num * 10 + s.charAt(i) - '0';
            }
            if ((!Character.isDigit(s.charAt(i)) && ' ' != s.charAt(i)) || i == len - 1) {
                switch (sign) {
                    case '-':
                        stack.push(-num);
                        break;
                    case '+':
                        stack.push(num);
                        break;
                    case '*':
                        stack.push(stack.pop() * num);
                        break;
                    case '/':
                        stack.push(stack.pop() / num);
                        break;
                }
                sign = s.charAt(i);
                num = 0;
            }
        }
        int result = 0;
        for (int i : stack) {
            result += i;
        }
        return result;
    }

    public int kthSmallest(TreeNode root, int k) {
        Deque<TreeNode> stack = new ArrayDeque<>();

        while (true) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (--k == 0) {
                return root.val;
            }
            root = root.right;
        }
    }

}
