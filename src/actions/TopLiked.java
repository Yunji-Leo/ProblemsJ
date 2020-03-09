package actions;

import actions.util.ListNode;
import actions.util.TreeNode;

import java.util.*;

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

    public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }
        if (target < nums[0] || target > nums[nums.length - 1]) {
            return new int[]{-1, -1};
        }

        int[] result = new int[2];
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = (left + right) / 2;
            if (nums[mid] >= target) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        if (nums[left] == target) {
            result[0] = left;
        } else {
            return new int[]{-1, -1};
        }

        left = 0;
        right = nums.length - 1;
        while (left < right - 1) {
            int mid = (left + right) / 2;
            if (nums[mid] <= target) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        if (nums[right] == target) {
            result[1] = right;
        } else {
            result[1] = left;
        }


        return result;
    }

    public int trap(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }

        int[] leftMax = new int[height.length];
        int[] rightMax = new int[height.length];
        int currMax = 0;
        for (int i = 0; i < height.length; i++) {
            currMax = Math.max(currMax, height[i]);
            if (currMax > height[i]) {
                leftMax[i] = currMax - height[i];
            }
        }
        currMax = 0;
        for (int i = height.length - 1; i >= 0; i--) {
            currMax = Math.max(currMax, height[i]);
            if (currMax > height[i]) {
                rightMax[i] = currMax - height[i];
            }
        }

        int result = 0;
        for (int i = 0; i < height.length; i++) {
            result += Math.min(leftMax[i], rightMax[i]);
        }
        return result;
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        permuteBacktrack(list, new ArrayList<>(), nums);
        return list;
    }

    private void permuteBacktrack(List<List<Integer>> list, List<Integer> temp, int[] nums) {
        if (temp.size() == nums.length) {
            list.add(new ArrayList<>(temp));
        } else {
            for (int i = 0; i < nums.length; i++) {
                if (temp.contains(nums[i])) continue;
                temp.add(nums[i]);
                permuteBacktrack(list, temp, nums);
                temp.remove(temp.size() - 1);
            }
        }

    }

    public void rotate(int[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = i; j < matrix[0].length; j++) {
                int temp = 0;
                temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length / 2; j++) {
                int temp = 0;
                temp = matrix[i][j];
                matrix[i][j] = matrix[i][matrix.length - 1 - j];
                matrix[i][matrix.length - 1 - j] = temp;
            }
        }
    }

    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[m - 1][n - 1] = grid[m - 1][n - 1];
        for (int i = m - 2; i >= 0; i--) {
            dp[i][n - 1] = grid[i][n - 1] + dp[i + 1][n - 1];
        }
        for (int j = n - 2; j >= 0; j--) {
            dp[m - 1][j] = grid[m - 1][j] + dp[m - 1][j + 1];
        }
        for (int i = m - 2; i >= 0; i--)
            for (int j = n - 2; j >= 0; j--) {
                dp[i][j] = grid[i][j] + Math.min(dp[i + 1][j], dp[i][j + 1]);
            }
        return dp[0][0];
    }

    public void sortColors(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        int mid = 0;
        while (mid < right) {
            if (nums[mid] == 0) {
                if (mid == left) {
                    mid++;
                    left++;
                } else {
                    swap(nums, mid, left);
                    left++;
                }
            } else if (nums[mid] == 2) {
                swap(nums, mid, right);
                right--;
            } else {
                mid++;
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        subsetsBacktrack(nums, 0, result, new ArrayList<>());
        return result;
    }

    private void subsetsBacktrack(int[] nums, int index, List<List<Integer>> result, List<Integer> temp) {
        result.add(new ArrayList<>(temp));

        for (int i = index; i < nums.length; i++) {
            temp.add(nums[i]);
            subsetsBacktrack(nums, i + 1, result, temp);
            temp.remove(temp.size() - 1);
        }
    }

    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0) {
            return word == null || word.length() == 0;
        }

        int m = board.length;
        int n = board[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (existBacktrack(board, word, visited, i, j, 0))
                    return true;
            }
        }
        return false;
    }

    private boolean existBacktrack(char[][] board, String word, boolean[][] visited, int i, int j, int index) {
        if (index == word.length())
            return true;
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || visited[i][j] == true)
            return false;
        if (board[i][j] != word.charAt(index))
            return false;

        int[] dr = new int[]{1, 0, -1, 0};
        int[] dc = new int[]{0, 1, 0, -1};
        visited[i][j] = true;
        for (int k = 0; k < 4; k++) {
            if (existBacktrack(board, word, visited, i + dr[k], j + dr[k], index + 1))
                return true;
        }
        visited[i][j] = false;
        return false;
    }

    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int maxArea = 0;
        int[] height = new int[matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == '0') {
                    height[j] = 0;
                } else {
                    height[j] = 1 + height[j];
                }
            }
            maxArea = Math.max(maxArea, calculateMaxArea(height));
        }
        return maxArea;
    }

    private int calculateMaxArea(int[] height) {
        int result = 0;
        int[] leftMax = new int[height.length];
        int[] rightMax = new int[height.length];
        leftMax[0] = -1;
        for (int i = 1; i < height.length; i++) {
            int p = i - 1;
            while (p >= 0 && height[i] <= height[p]) {
                p = leftMax[p];
            }
            leftMax[i] = p;
        }
        rightMax[rightMax.length - 1] = rightMax.length;
        for (int i = rightMax.length - 2; i >= 0; i--) {
            int p = i + 1;
            while (p < rightMax.length && height[i] <= height[p]) {
                p = rightMax[p];
            }
            rightMax[i] = p;
        }
        for (int i = 0; i < height.length; i++) {
            result = Math.max(result, height[i] * (rightMax[i] - leftMax[i] - 1));
        }
        return result;
    }

    HashMap<Integer, Integer> numTreesMap = new HashMap<>();

    public int numTrees(int n) {
        if (n == 1 || n == 0) {
            return 1;
        }
        if (numTreesMap.containsKey(n)) {
            return numTreesMap.get(n);
        }
        int result = 0;
        for (int i = 1; i <= n; i++) {
            result += numTrees(i - 1) * numTrees(n - i - 1);
        }
        numTreesMap.put(n, result);
        return result;
    }

    public boolean isValidBST(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        double inorder = -Double.MAX_VALUE;
        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (root.val <= inorder) return false;
            inorder = root.val;
            root = root.right;
        }
        return true;
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isSymmetricRecur(root.left, root.right);
    }

    private boolean isSymmetricRecur(TreeNode leftRoot, TreeNode rightRoot) {
        if (leftRoot == null && rightRoot == null) {
            return true;
        }
        if (leftRoot == null || rightRoot == null) {
            return false;
        }
        if (leftRoot.val != rightRoot.val) {
            return false;
        }
        return isSymmetricRecur(leftRoot.left, rightRoot.right) && isSymmetricRecur(leftRoot.right, rightRoot.left);
    }

    public int singleNumber(int[] nums) {
        int result = 0;
        for (int n : nums) {
            result ^= n;
        }
        return result;
    }

    HashSet<String> wordBreakSet = new HashSet<>();

    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || s.length() == 0) {
            return true;
        }
        if (wordBreakSet.contains(s)) {
            return false;
        }

        for (String word : wordDict) {
            if (s.startsWith(word)) {
                if (wordBreak(s.substring(word.length()), wordDict)) {
                    return true;
                }
            }
        }
        wordBreakSet.add(s);
        return false;
    }

    public ListNode detectCycle(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                break;
            }
        }
        if (fast == null || fast.next == null) {
            return null;
        }
        fast = head;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }

    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode prev = null, slow = head, fast = head;

        while (fast != null && fast.next != null) {
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }

        prev.next = null;

        ListNode l1 = sortList(head);
        ListNode l2 = sortList(slow);

        return merge(l1, l2);
    }

    ListNode merge(ListNode l1, ListNode l2) {
        ListNode l = new ListNode(0), p = l;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                p.next = l1;
                l1 = l1.next;
            } else {
                p.next = l2;
                l2 = l2.next;
            }
            p = p.next;
        }
        if (l1 != null)
            p.next = l1;
        if (l2 != null)
            p.next = l2;
        return l.next;
    }

    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }

        int m = matrix.length;
        int n = matrix[0].length;
        int[][] dp = new int[m][n];
        int result = 0;
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == '1') {
                dp[i][0] = 1;
                result = 1;
            }
        }
        for (int j = 0; j < n; j++) {
            if (matrix[0][j] == '1') {
                dp[0][j] = 1;
                result = 1;
            }
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == '1') {
                    dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
                    result = Math.max(result, dp[i][j]);
                }
            }
        }
        return result * result;
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return root;
        }
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }

    public int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] answer = new int[length];
        answer[0] = 1;
        for (int i = 1; i < length; i++) {
            answer[i] = nums[i - 1] * answer[i - 1];
        }
        int R = 1;
        for (int i = length - 1; i >= 0; i--) {
            answer[i] = answer[i] * R;
            R *= nums[i];
        }
        return answer;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length < 1 || matrix[0].length < 1) {
            return false;
        }
        int col = matrix[0].length - 1;
        int row = 0;
        while (col >= 0 && row <= matrix.length - 1) {
            if (target == matrix[row][col]) {
                return true;
            } else if (target < matrix[row][col]) {
                col--;
            } else if (target > matrix[row][col]) {
                row++;
            }
        }
        return false;
    }

    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            int minNum = Integer.MAX_VALUE;
            int k = 1;
            while (k * k <= i) {
                minNum = Math.min(minNum, dp[i - k * k] + 1);
                k++;
            }
            dp[i] = minNum;
        }
        return dp[n];
    }

    public void moveZeroes(int[] nums) {
        int slow = 0;
        int fast = 0;
        while (fast < nums.length) {
            if (nums[fast] != 0) {
                swap(nums, slow, fast);
                slow++;
            }
            fast++;
        }
    }

    public class Codec {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            Queue<TreeNode> queue = new LinkedList<>();
            queue.add(root);
            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();
                if (node == null) {
                    sb.append("null");
                } else {
                    sb.append(node.val);
                    queue.add(node.left);
                    queue.add(node.right);
                }
                sb.append(",");
            }
            return sb.toString();
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            String[] fields = data.split(",");
            if (fields == null || fields.length == 0 || fields[0].equals("null")) {
                return null;
            }
            Queue<TreeNode> queue = new LinkedList<>();
            int i = 0;
            TreeNode root = new TreeNode(Integer.valueOf(fields[i]));
            queue.add(root);
            i++;
            while (!queue.isEmpty()) {
                TreeNode curr = queue.poll();
                if (!fields[i].equals("null")) {
                    TreeNode left = new TreeNode(Integer.valueOf(fields[i]));
                    curr.left = left;
                    queue.add(left);
                }
                i++;
                if (!fields[i].equals("null")) {
                    TreeNode right = new TreeNode(Integer.valueOf(fields[i]));
                    curr.right = right;
                    queue.add(right);
                }
                i++;
            }
            return root;
        }
    }

    public String decodeString(String s) {
        Stack<Integer> numStack = new Stack<>();
        Stack<String> strStack = new Stack<>();
        String curr = "";
        int i = 0;
        while (i < s.length()) {
            if (Character.isDigit(s.charAt(i))) {
                //strStack.push(curr);
                int num = 0;
                while (Character.isDigit(s.charAt(i))) {
                    num = num * 10 + s.charAt(i) - '0';
                    i++;
                }
                numStack.push(num);
            } else if (s.charAt(i) == '[') {
                strStack.push(curr);
                curr = "";
                i++;
            } else if (s.charAt(i) == ']') {
                int count = numStack.pop();
                StringBuilder res = new StringBuilder(strStack.pop());
                while (count > 0) {
                    res.append(curr);
                    count--;
                }
                curr = res.toString();
                i++;
            } else {
                curr += s.charAt(i);
                i++;
            }
        }
        return curr;
    }

    public int[][] reconstructQueue(int[][] people) {
        if (people == null || people.length == 0 || people[0].length == 0)
            return new int[0][0];

        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return o1[1] - o2[1];
                }
                return o2[0] - o1[0];
            }
        });

        int n = people.length;
        ArrayList<int[]> tmp = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            tmp.add(people[i][1], new int[]{people[i][0], people[i][1]});
        }

        int[][] res = new int[people.length][2];
        int i = 0;
        for (int[] k : tmp) {
            res[i][0] = k[0];
            res[i][1] = k[1];
            i++;
        }
        return res;
    }

    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum % 2 == 1) {
            return false;
        }
        sum /= 2;
        int n = nums.length;
        boolean[][] dp = new boolean[n + 1][sum + 1];
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            dp[i][0] = true;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= sum; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= nums[i - 1]) {
                    dp[i][j] = (dp[i][j] || dp[i - 1][j - nums[i - 1]]);
                }
            }
        }
        return dp[n][sum];
    }

}
